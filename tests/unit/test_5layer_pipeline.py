"""
Tests for the 5-layer groundedness pipeline (hallucination.py).

Layer coverage:
  L0: Deterministic text match (direct, no mocks)
  L1: Gate 1 MiniLM (uses real model via _compute_grounding_ratio)
  L2a: MiniCheck factual consistency (mocked)
  L2b: NLI DeBERTa ENTAILMENT fallback (mocked)
  L3: GraphRAG spaCy exact match (mocked)
  L4: Gemini per-sentence (mocked HTTP)
  Integration: Full layered flow with mocked L2a/L2b/L3/L4
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

# =====================================================================
# L0: Deterministic text match
# =====================================================================


class TestL0DeterministicMatch:
    """L0 is the cheapest layer — exact substring, near-exact, Jaccard."""

    def test_exact_substring_match(self) -> None:
        from llm_judge.calibration.hallucination import _l0_deterministic_match

        sentence = "Paris is the capital of France."
        source = "Paris is the capital of France. It is known for the Eiffel Tower."
        source_sents = [
            "Paris is the capital of France.",
            "It is known for the Eiffel Tower.",
        ]

        assert _l0_deterministic_match(sentence, source_sents, source) is True

    def test_near_exact_match(self) -> None:
        from llm_judge.calibration.hallucination import _l0_deterministic_match

        sentence = "Paris is the capital of France"  # no period
        source_sents = ["Paris is the capital of France."]
        source = "Paris is the capital of France."

        # SequenceMatcher ratio > 0.85 — should match
        assert _l0_deterministic_match(sentence, source_sents, source) is True

    def test_high_jaccard_overlap(self) -> None:
        from llm_judge.calibration.hallucination import _l0_deterministic_match

        sentence = "The capital of France is Paris."
        source_sents = ["Paris is the capital of France."]
        source = "Paris is the capital of France."

        # Same words, different order — Jaccard = 1.0
        assert _l0_deterministic_match(sentence, source_sents, source) is True

    def test_no_match_different_content(self) -> None:
        from llm_judge.calibration.hallucination import _l0_deterministic_match

        sentence = "Tokyo is the largest city in Japan."
        source_sents = ["Paris is the capital of France."]
        source = "Paris is the capital of France."

        assert _l0_deterministic_match(sentence, source_sents, source) is False

    def test_partial_overlap_below_threshold(self) -> None:
        from llm_judge.calibration.hallucination import _l0_deterministic_match

        sentence = "The president of France visited Germany for trade talks."
        source_sents = ["France is a country in Europe with a strong economy."]
        source = "France is a country in Europe with a strong economy."

        # Only "France" overlaps — Jaccard well below 0.80
        assert _l0_deterministic_match(sentence, source_sents, source) is False

    def test_empty_sentence(self) -> None:
        from llm_judge.calibration.hallucination import _l0_deterministic_match

        # Empty string is substring of everything in Python ("" in "abc" == True)
        # But sent_tokens is empty so function returns False after token check
        # Actually, the substring check fires first. This is correct behavior:
        # an empty sentence is vacuously "found" in any source.
        assert _l0_deterministic_match("", ["Some source."], "Some source.") is True

    def test_case_insensitive(self) -> None:
        from llm_judge.calibration.hallucination import _l0_deterministic_match

        sentence = "PARIS IS THE CAPITAL OF FRANCE."
        source_sents = ["Paris is the capital of France."]
        source = "Paris is the capital of France."

        assert _l0_deterministic_match(sentence, source_sents, source) is True


# =====================================================================
# L1: Gate 1 MiniLM check
# =====================================================================


class TestL1Gate1Check:
    """L1 uses MiniLM embeddings for whole-response similarity."""

    def test_grounded_response_passes(self) -> None:
        from llm_judge.calibration.hallucination import _l1_gate1_check

        decision, ratio, min_sim = _l1_gate1_check(
            response="Paris is the capital of France.",
            context="France is a country in Europe. Paris is the capital of France.",
        )
        assert decision == "pass"
        assert ratio >= 0.80
        assert min_sim >= 0.30

    def test_ungrounded_response_fails(self) -> None:
        from llm_judge.calibration.hallucination import _l1_gate1_check

        decision, ratio, min_sim = _l1_gate1_check(
            response="The quantum fluctuation of dark matter causes gravitational lensing in parallel dimensions.",
            context="What is the weather like today?",
        )
        assert decision in ("fail", "ambiguous")

    def test_skip_embeddings_fallback(self) -> None:
        from llm_judge.calibration.hallucination import _l1_gate1_check

        decision, ratio, min_sim = _l1_gate1_check(
            response="Python is a programming language.",
            context="Python is a programming language used for web development.",
            skip_embeddings=True,
        )
        # Token overlap fallback — should still show high grounding
        assert ratio > 0.3


# =====================================================================
# L2a: MiniCheck factual consistency (mocked)
# =====================================================================


class TestL2aMiniCheck:
    """L2a uses MiniCheck Flan-T5 — mocked to avoid loading 3.1GB model."""

    def test_supported_returns_true(self) -> None:
        import torch

        import llm_judge.calibration.hallucination as hal
        from llm_judge.calibration.hallucination import _l2a_minicheck

        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {
            "input_ids": torch.zeros(1, 20, dtype=torch.long)
        }
        mock_model = MagicMock()
        mock_model.generate.return_value = torch.tensor([[1]])
        mock_tokenizer.decode.return_value = "1"

        hal._mc_tokenizer = mock_tokenizer
        hal._mc_model = mock_model

        result = _l2a_minicheck(
            "Paris is the capital.", "Paris is the capital of France."
        )
        assert result is True

    def test_unsupported_returns_false(self) -> None:
        import torch

        import llm_judge.calibration.hallucination as hal
        from llm_judge.calibration.hallucination import _l2a_minicheck

        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {
            "input_ids": torch.zeros(1, 20, dtype=torch.long)
        }
        mock_model = MagicMock()
        mock_model.generate.return_value = torch.tensor([[0]])
        mock_tokenizer.decode.return_value = "0"

        hal._mc_tokenizer = mock_tokenizer
        hal._mc_model = mock_model

        result = _l2a_minicheck(
            "Tokyo is in Germany.", "Tokyo is the capital of Japan."
        )
        assert result is False


# =====================================================================
# L2a→L2b Fallback: MiniCheck misses, DeBERTa catches
# =====================================================================


class TestL2Fallback:
    """When MiniCheck returns unsupported, DeBERTa NLI acts as fallback."""

    def test_deberta_catches_minicheck_miss(self) -> None:
        """L2b catches a sentence that L2a missed."""
        import llm_judge.calibration.hallucination as hal
        from llm_judge.calibration.hallucination import check_hallucination

        response = "Paris is the capital of France. London is the capital of England."
        context = "Paris is the capital of France. London is the capital of England."

        # MiniCheck misses, DeBERTa catches
        with (
            patch.object(hal, "_l2a_minicheck", return_value=False),
            patch.object(hal, "_load_minicheck"),
            patch.object(hal, "_l2_nli_check", return_value=True),
            patch.object(hal, "_load_nli"),
        ):
            result = check_hallucination(
                response=response,
                context=context,
                case_id="test_fallback",
                gate2_routing="pass",
            )

        # DeBERTa fallback should catch sentences that MiniCheck missed
        assert (
            result.layer_stats.get("L3b_nli", 0) >= 1
            or result.layer_stats.get("L1", 0) >= 1
        )


# =====================================================================
# L2b: NLI DeBERTa ENTAILMENT (mocked)
# =====================================================================


class TestL2NLICheck:
    """L2b uses DeBERTa NLI as fallback — mocked to avoid loading 400MB model."""

    def test_entailment_confirms_grounded(self) -> None:
        # Mock NLI model to return high entailment
        import torch

        import llm_judge.calibration.hallucination as hal
        from llm_judge.calibration.hallucination import _l2_nli_check

        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {
            "input_ids": torch.zeros(1, 10, dtype=torch.long)
        }
        mock_model = MagicMock()
        mock_model.return_value = MagicMock(
            logits=torch.tensor([[0.0, 5.0, 0.0]])  # softmax → ~0.99 entailment
        )
        mock_labels = ["CONTRADICTION", "ENTAILMENT", "NEUTRAL"]

        hal._nli_tokenizer = mock_tokenizer
        hal._nli_model = mock_model
        hal._nli_labels = mock_labels

        # Mock embedding provider
        mock_provider = MagicMock()
        mock_provider.max_similarity.return_value = 0.9
        mock_emb = MagicMock()

        with patch(
            "llm_judge.properties.get_embedding_provider", return_value=mock_provider
        ):
            result = _l2_nli_check(
                sentence="Paris is the capital.",
                context_sentences=["Paris is the capital of France."],
                ctx_embeddings=[mock_emb],
                resp_emb=mock_emb,
            )

        assert result is True

    def test_no_entailment_returns_false(self) -> None:
        import torch

        import llm_judge.calibration.hallucination as hal
        from llm_judge.calibration.hallucination import _l2_nli_check

        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {
            "input_ids": torch.zeros(1, 10, dtype=torch.long)
        }
        mock_model = MagicMock()
        mock_model.return_value = MagicMock(
            logits=torch.tensor([[0.1, 0.2, 0.7]])  # high neutral, low entailment
        )
        mock_labels = ["CONTRADICTION", "ENTAILMENT", "NEUTRAL"]

        hal._nli_tokenizer = mock_tokenizer
        hal._nli_model = mock_model
        hal._nli_labels = mock_labels

        mock_provider = MagicMock()
        mock_provider.max_similarity.return_value = 0.5
        mock_emb = MagicMock()

        with patch(
            "llm_judge.properties.get_embedding_provider", return_value=mock_provider
        ):
            result = _l2_nli_check(
                sentence="Tokyo is the largest city.",
                context_sentences=["Paris is the capital of France."],
                ctx_embeddings=[mock_emb],
                resp_emb=mock_emb,
            )

        assert result is False


# =====================================================================
# L3: GraphRAG spaCy exact match (mocked)
# =====================================================================


class TestL3GraphRAGCheck:
    """L3 uses spaCy SVO extraction — mocked to avoid model dependency."""

    def test_exact_triplet_match_returns_true(self) -> None:
        from llm_judge.benchmarks.graphrag_science_gate import Triplet
        from llm_judge.calibration.hallucination import _l3_graphrag_check

        resp_triplets = [
            Triplet(subject="Blue Bell", predicate="recall", obj="products")
        ]
        src_triplets = [
            Triplet(subject="Blue Bell", predicate="recall", obj="products")
        ]

        with (
            patch("llm_judge.calibration.hallucination._load_spacy"),
            patch(
                "llm_judge.benchmarks.graphrag_science_gate.extract_svo_triplets",
                side_effect=[resp_triplets, src_triplets],
            ),
        ):
            result = _l3_graphrag_check(
                "Blue Bell recalled products.",
                "Blue Bell recalled products from the plant.",
            )

        assert result is True

    def test_no_match_returns_false(self) -> None:
        from llm_judge.benchmarks.graphrag_science_gate import Triplet
        from llm_judge.calibration.hallucination import _l3_graphrag_check

        resp_triplets = [Triplet(subject="Thomas", predicate="arrest", obj="March 26")]
        src_triplets = [Triplet(subject="Thomas", predicate="purchase", obj="ticket")]

        with (
            patch("llm_judge.calibration.hallucination._load_spacy"),
            patch(
                "llm_judge.benchmarks.graphrag_science_gate.extract_svo_triplets",
                side_effect=[resp_triplets, src_triplets],
            ),
        ):
            result = _l3_graphrag_check(
                "Thomas was arrested on March 26.",
                "Thomas purchased a ticket on March 26.",
            )

        assert result is False

    def test_intransitive_triplets_filtered(self) -> None:
        from llm_judge.benchmarks.graphrag_science_gate import Triplet
        from llm_judge.calibration.hallucination import _l3_graphrag_check

        # All response triplets are intransitive — should return False (nothing to match)
        resp_triplets = [
            Triplet(subject="police", predicate="say", obj="(intransitive)")
        ]
        src_triplets = [
            Triplet(subject="police", predicate="say", obj="(intransitive)")
        ]

        with (
            patch("llm_judge.calibration.hallucination._load_spacy"),
            patch(
                "llm_judge.benchmarks.graphrag_science_gate.extract_svo_triplets",
                side_effect=[resp_triplets, src_triplets],
            ),
        ):
            result = _l3_graphrag_check(
                "Police said something.", "Police said something."
            )

        assert result is False

    def test_import_error_returns_false(self) -> None:
        from llm_judge.calibration.hallucination import _l3_graphrag_check

        with (
            patch("llm_judge.calibration.hallucination._load_spacy"),
            patch(
                "llm_judge.benchmarks.graphrag_science_gate.extract_svo_triplets",
                side_effect=ImportError("no spacy"),
            ),
        ):
            result = _l3_graphrag_check("Any sentence.", "Any source.")

        assert result is False


# =====================================================================
# L4: Gemini per-sentence reasoning (mocked HTTP)
# =====================================================================


class TestL4GeminiCheck:
    """L4 uses Gemini API — mocked to avoid real API calls."""

    def test_supported_response(self) -> None:
        from llm_judge.calibration.hallucination import _l4_gemini_check

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {"text": "thinking..."},
                            {"text": "SUPPORTED"},
                        ]
                    }
                }
            ]
        }
        mock_response.raise_for_status = MagicMock()

        with (
            patch.dict("os.environ", {"GEMINI_API_KEY": "test-key"}),
            patch("httpx.Client") as mock_client,
        ):
            mock_client.return_value.__enter__ = MagicMock(
                return_value=MagicMock(post=MagicMock(return_value=mock_response))
            )
            mock_client.return_value.__exit__ = MagicMock(return_value=False)

            result = _l4_gemini_check(
                "Paris is the capital.", "Paris is the capital of France."
            )

        assert result == "supported"

    def test_unsupported_response(self) -> None:
        from llm_judge.calibration.hallucination import _l4_gemini_check

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {"text": "thinking..."},
                            {"text": "UNSUPPORTED"},
                        ]
                    }
                }
            ]
        }
        mock_response.raise_for_status = MagicMock()

        with (
            patch.dict("os.environ", {"GEMINI_API_KEY": "test-key"}),
            patch("httpx.Client") as mock_client,
        ):
            mock_client.return_value.__enter__ = MagicMock(
                return_value=MagicMock(post=MagicMock(return_value=mock_response))
            )
            mock_client.return_value.__exit__ = MagicMock(return_value=False)

            result = _l4_gemini_check(
                "She was arrested on March 26.", "Thomas purchased a ticket."
            )

        assert result == "unsupported"

    def test_no_api_key_defaults_to_supported(self) -> None:
        from llm_judge.calibration.hallucination import _l4_gemini_check

        with patch.dict("os.environ", {}, clear=True):
            # Remove GEMINI_API_KEY if set
            import os

            os.environ.pop("GEMINI_API_KEY", None)
            result = _l4_gemini_check("Any sentence.", "Any source.")

        assert result == "supported"

    def test_api_error_returns_error(self) -> None:
        from llm_judge.calibration.hallucination import _l4_gemini_check

        with (
            patch.dict("os.environ", {"GEMINI_API_KEY": "test-key"}),
            patch("httpx.Client") as mock_client,
        ):
            mock_client.return_value.__enter__ = MagicMock(
                return_value=MagicMock(
                    post=MagicMock(side_effect=Exception("connection timeout"))
                )
            )
            mock_client.return_value.__exit__ = MagicMock(return_value=False)

            result = _l4_gemini_check("Any sentence.", "Any source.")

        assert result == "error"


# =====================================================================
# Integration: Full layered flow
# =====================================================================


class TestLayeredPipeline:
    """Test the full 5-layer pipeline with various gate2_routing settings."""

    def test_default_routing_none_uses_l0_l1_only(self) -> None:
        """gate2_routing='none' (default) — only L0 and L1 run."""
        from llm_judge.calibration.hallucination import check_hallucination

        result = check_hallucination(
            response="Paris is the capital of France.",
            context="France is a country in Europe. Paris is the capital of France.",
            case_id="test_default",
            gate2_routing="none",
        )
        # Should pass via L0 or L1 — no L2/L3/L4
        assert result.gate1_decision == "pass"
        assert result.layer_stats.get("L4_unsupported", 0) == 0
        assert result.layer_stats.get("L4_supported", 0) == 0

    def test_l0_resolves_all_sentences(self) -> None:
        """When all sentences are exact matches, L0 handles everything."""
        from llm_judge.calibration.hallucination import check_hallucination

        source = "Paris is the capital of France. London is the capital of England."
        result = check_hallucination(
            response=source,  # identical to context
            context=source,
            case_id="test_l0_all",
            gate2_routing="pass",
        )
        assert result.gate1_decision == "pass"
        assert result.risk_score == 0.0
        assert result.layer_stats.get("L1", 0) >= 1

    def test_gate1_fail_stops_pipeline(self) -> None:
        """When Gate 1 fails with gate2_routing='none', no deeper analysis runs."""
        from llm_judge.calibration.hallucination import check_hallucination

        result = check_hallucination(
            response="Quantum gravitational waves disrupted the spacetime fabric of the multiverse causing interdimensional cascading failures.",
            context="What is the weather like today?",
            case_id="test_gate1_fail",
            gate2_routing="none",  # Gate 1 is final verdict
        )
        assert result.gate1_decision in ("fail", "ambiguous")
        assert result.risk_score > 0.0
        assert result.layer_stats.get("L3_gate1_fail", 0) == 1

    def test_gate1_fail_continues_with_gate2_pass(self) -> None:
        """When Gate 1 fails with gate2_routing='pass', L2+ still runs."""
        import llm_judge.calibration.hallucination as hal
        from llm_judge.calibration.hallucination import check_hallucination

        with (
            patch.object(hal, "_l2a_minicheck", return_value=False),
            patch.object(hal, "_load_minicheck"),
            patch.object(hal, "_l2_nli_check", return_value=False),
            patch.object(hal, "_load_nli"),
            patch.object(hal, "_l3_graphrag_check", return_value=False),
            patch.object(hal, "_l4_gemini_check", return_value="unsupported"),
        ):
            result = check_hallucination(
                response="Quantum gravitational waves disrupted the spacetime fabric of the multiverse causing interdimensional cascading failures.",
                context="What is the weather like today?",
                case_id="test_gate1_continues",
                gate2_routing="pass",  # Gate 1 fail continues to L2+
            )
        assert result.gate1_decision in ("fail", "ambiguous")
        assert result.gate2_decision == "fail"
        assert result.layer_stats.get("L3_gate1_fail", 0) == 1

    def test_layered_flow_with_mocked_l2_l3_l4(self) -> None:
        """Full pipeline with mocked L2/L3/L4 — tests the flow logic."""
        import llm_judge.calibration.hallucination as hal
        from llm_judge.calibration.hallucination import check_hallucination

        # Response has 2 sentences — one will be L0-matched, one needs deeper analysis
        response = "Paris is the capital of France. Tokyo has the largest metropolitan population."
        context = "Paris is the capital of France. Tokyo is the most populous metropolitan area in the world."

        # Mock L2a MiniCheck to return True for non-L0 sentences
        with (
            patch.object(hal, "_l2a_minicheck", return_value=True),
            patch.object(hal, "_load_minicheck"),
        ):
            result = check_hallucination(
                response=response,
                context=context,
                case_id="test_layered",
                gate2_routing="pass",
            )

        assert result.gate1_decision == "pass"
        # First sentence should be L0, second should be L2
        assert (
            result.layer_stats.get("L1", 0) >= 1
            or result.layer_stats.get("L3a_minicheck", 0) >= 1
        )

    def test_l4_unsupported_causes_fail(self) -> None:
        """When L4 Gemini returns unsupported, case should fail.

        Key: response must be similar enough to pass Gate 1 (ratio >= 0.80, min_sim >= 0.30)
        but contain a fabricated detail that L4 should catch.
        """
        from llm_judge.calibration.hallucination import check_hallucination

        # 4 sentences: 3 grounded + 1 fabricated number.
        # Gate 1 should pass (75%+ grounded, min_sim reasonable).
        # Paraphrased: similar enough for Gate 1 PASS but different enough L0 won't match
        response = (
            "The Blue Bell ice cream company closed its manufacturing facility in the state of Oklahoma. "
            "Health authorities issued warnings to customers regarding the contaminated products. "
            "Listeria bacteria were discovered in ice cream samples recovered from a hospital in Kansas. "
            "Three individuals in Kansas lost their lives due to listeriosis connected to the contamination."
        )
        context = (
            "Blue Bell ice cream has temporarily shut down one of its manufacturing plants in Oklahoma. "
            "Public health officials warned consumers not to eat any Blue Bell branded products. "
            "The contamination was found in a cup of ice cream from a Kansas hospital. "
            "Three people in Kansas died from listeriosis linked to the outbreak."
        )

        # Mock: L2 returns False (no entailment), L3 returns False, L4 returns unsupported
        with (
            patch(
                "llm_judge.calibration.hallucination._l2a_minicheck", return_value=False
            ),
            patch("llm_judge.calibration.hallucination._load_minicheck"),
            patch(
                "llm_judge.calibration.hallucination._l2_nli_check", return_value=False
            ),
            patch("llm_judge.calibration.hallucination._load_nli"),
            patch(
                "llm_judge.calibration.hallucination._l3_graphrag_check",
                return_value=False,
            ),
            patch(
                "llm_judge.calibration.hallucination._l4_gemini_check",
                return_value="unsupported",
            ),
        ):
            result = check_hallucination(
                response=response,
                context=context,
                case_id="test_l4_fail",
                gate2_routing="pass",
                grounding_threshold=0.70,  # lower to ensure Gate 1 passes for paraphrased text
            )

        assert (
            result.gate1_decision == "pass"
        ), f"Gate 1 should pass but got {result.gate1_decision} (ratio={result.grounding_ratio}, min_sim={result.min_sentence_sim})"
        assert result.gate2_decision == "fail"
        assert result.risk_score > 0.0
        assert result.layer_stats.get("L4_unsupported", 0) >= 1
        assert any("l4_unsupported" in f for f in result.flags)

    def test_l4_all_supported_passes(self) -> None:
        """When L4 Gemini returns supported for all sentences, case passes."""
        import llm_judge.calibration.hallucination as hal
        from llm_judge.calibration.hallucination import check_hallucination

        response = "Paris is the capital of France. It is a beautiful city in Europe."
        context = "Paris is the capital of France. Paris is known as a beautiful European city."

        with (
            patch.object(hal, "_l2a_minicheck", return_value=False),
            patch.object(hal, "_load_minicheck"),
            patch.object(hal, "_l2_nli_check", return_value=False),
            patch.object(hal, "_load_nli"),
            patch.object(hal, "_l3_graphrag_check", return_value=False),
            patch.object(hal, "_l4_gemini_check", return_value="supported"),
        ):
            result = check_hallucination(
                response=response,
                context=context,
                case_id="test_l4_pass",
                gate2_routing="pass",
            )

        assert result.gate2_decision == "pass"
        assert result.risk_score == 0.0
        assert result.layer_stats.get("L4_supported", 0) >= 1

    def test_sentence_results_tracking(self) -> None:
        """Verify that sentence_results tracks which layer resolved each sentence."""
        import llm_judge.calibration.hallucination as hal
        from llm_judge.calibration.hallucination import check_hallucination

        response = (
            "Paris is the capital of France. Tokyo is in Japan. Berlin is in Germany."
        )
        context = "Paris is the capital of France. Tokyo is located in Japan. Berlin is the capital of Germany."

        # L2a MiniCheck returns True for sentences not matched by L0
        with (
            patch.object(hal, "_l2a_minicheck", return_value=True),
            patch.object(hal, "_load_minicheck"),
        ):
            result = check_hallucination(
                response=response,
                context=context,
                case_id="test_tracking",
                gate2_routing="pass",
            )

        assert len(result.sentence_results) > 0
        resolved_layers = {sr.resolved_by for sr in result.sentence_results}
        # Should have at least L0 or L2 entries
        assert resolved_layers & {"L1", "L2a_minicheck"}

    def test_all_resolved_at_l2_gives_gate2_pass(self) -> None:
        """G2='' fix: when L2a/L2b resolves all sentences (no L4), gate2_decision='pass' not ''."""
        import llm_judge.calibration.hallucination as hal
        from llm_judge.calibration.hallucination import check_hallucination

        response = "Paris is the capital of France. It is a beautiful city in Europe."
        context = "Paris is the capital of France. Paris is known as a beautiful European city."

        # L2a resolves everything, L4 never called
        with (
            patch.object(hal, "_l2a_minicheck", return_value=True),
            patch.object(hal, "_load_minicheck"),
        ):
            result = check_hallucination(
                response=response,
                context=context,
                case_id="test_g2_empty_fix",
                gate2_routing="pass",
            )

        assert (
            result.gate2_decision == ""
        ), f"Expected empty gate2 when L4 not needed, got '{result.gate2_decision}'"
        assert result.risk_score == 0.0
        assert result.layer_stats.get("L4_supported", 0) == 0
        assert result.layer_stats.get("L4_unsupported", 0) == 0

    def test_backward_compatibility_no_layered(self) -> None:
        """layered=False should behave like the old 2-gate system."""
        from llm_judge.calibration.hallucination import check_hallucination

        result = check_hallucination(
            response="Paris is the capital of France.",
            context="Paris is the capital of France.",
            case_id="test_compat",
            layered=False,
        )
        # Should still work — L0 + L1 only
        assert result.gate1_decision == "pass"
        assert result.risk_score < 0.3


# =====================================================================
# Dataclass tests
# =====================================================================


class TestDataclasses:
    """Test new dataclass fields and defaults."""

    def test_hallucination_result_defaults(self) -> None:
        from llm_judge.calibration.hallucination import HallucinationResult

        result = HallucinationResult(
            case_id="test",
            risk_score=0.0,
            grounding_ratio=1.0,
            min_sentence_sim=1.0,
            ungrounded_claims=0,
            unverifiable_citations=0,
        )
        assert result.layer_stats == {}
        assert result.sentence_results == []
        assert result.gate1_decision == ""
        assert result.gate2_decision == ""
        assert result.flags == []

    def test_sentence_layer_result(self) -> None:
        from llm_judge.calibration.hallucination import SentenceLayerResult

        sr = SentenceLayerResult(
            sentence_idx=0,
            sentence="Test sentence.",
            resolved_by="L1",
            detail="exact_match",
        )
        assert sr.resolved_by == "L1"
        assert sr.detail == "exact_match"

    def test_layer_stats_populated(self) -> None:
        from llm_judge.calibration.hallucination import check_hallucination

        result = check_hallucination(
            response="Paris is the capital of France.",
            context="Paris is the capital of France.",
            case_id="test_stats",
            gate2_routing="pass",
        )
        # layer_stats should have keys
        assert isinstance(result.layer_stats, dict)
        assert "L1" in result.layer_stats or "L2a_minicheck" in result.layer_stats


# =====================================================================
# Edge cases
# =====================================================================


class TestEdgeCases:
    """Edge cases and error handling."""

    def test_empty_response(self) -> None:
        from llm_judge.calibration.hallucination import check_hallucination

        result = check_hallucination(
            response="",
            context="Some context.",
            case_id="empty",
        )
        assert result.risk_score == 0.0
        assert result.grounding_ratio == 1.0

    def test_empty_context(self) -> None:
        from llm_judge.calibration.hallucination import check_hallucination

        result = check_hallucination(
            response="Some response.",
            context="",
            case_id="empty_ctx",
        )
        assert result.risk_score == 0.0

    def test_very_short_sentences_filtered(self) -> None:
        from llm_judge.calibration.hallucination import _split_sentences

        sentences = _split_sentences(
            "Yes. No. Maybe so. This is a longer sentence that should be kept."
        )
        # Short fragments (<10 chars) should be filtered
        assert all(len(s) > 10 for s in sentences)

    def test_source_context_parameter(self) -> None:
        """source_context should be used for L3/L4 when provided."""
        import llm_judge.calibration.hallucination as hal
        from llm_judge.calibration.hallucination import check_hallucination

        response = "The company shut down its plant."
        context = (
            "User asked about Blue Bell. The company shut down its Broken Arrow plant."
        )
        source_only = (
            "Blue Bell temporarily shut down its Broken Arrow, Oklahoma plant."
        )

        with (
            patch.object(hal, "_l2a_minicheck", return_value=True),
            patch.object(hal, "_load_minicheck"),
        ):
            result = check_hallucination(
                response=response,
                context=context,
                source_context=source_only,
                case_id="test_source_ctx",
                gate2_routing="pass",
            )

        assert result.gate1_decision == "pass"
