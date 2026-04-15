"""
Tests for the 5th deep audit pass.

Covers:
  - AlphaModel bootstrap_priors includes legacy strategy aliases
"""


class TestAlphaModelLegacyAliasPriors:
    """Audit 5: alpha-model cold-start priors must honor legacy strategy aliases."""

    def test_bootstrap_priors_includes_legacy_aliases(self):
        from core.alpha_model import AlphaModel

        model = AlphaModel()
        model.bootstrap_priors()

        weights = model.get_all_weights()
        for alias in (
            "MomentumContinuation",
            "IchimokuCloud",
            "Wyckoff",
            "HarmonicDetector",
            "GeometricPatterns",
        ):
            assert alias in weights

    def test_legacy_aliases_match_canonical_priors(self):
        from core.alpha_model import AlphaModel

        model = AlphaModel()
        model.bootstrap_priors()

        pairs = [
            ("Momentum", "MomentumContinuation", 6.0, 5.0, 2.0, 1.0),
            ("Ichimoku", "IchimokuCloud", 5.0, 5.0, 1.8, 1.0),
            ("WyckoffAccDist", "Wyckoff", 6.0, 5.0, 2.1, 1.0),
            ("HarmonicPattern", "HarmonicDetector", 5.0, 5.0, 2.0, 1.0),
            ("GeometricPattern", "GeometricPatterns", 5.0, 5.0, 1.8, 1.0),
        ]

        for canonical, alias, alpha, beta, avg_win_r, avg_loss_r in pairs:
            canonical_sw = model._get_weight(canonical)
            alias_sw = model._get_weight(alias)

            assert canonical_sw.alpha == alpha
            assert canonical_sw.beta == beta
            assert canonical_sw.avg_win_r == avg_win_r
            assert canonical_sw.avg_loss_r == avg_loss_r

            assert alias_sw.alpha == canonical_sw.alpha
            assert alias_sw.beta == canonical_sw.beta
            assert alias_sw.avg_win_r == canonical_sw.avg_win_r
            assert alias_sw.avg_loss_r == canonical_sw.avg_loss_r
