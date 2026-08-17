"""Real-image asset pipeline for V3 synthetic documents.

Downloads license-clean real images (photos, portraits, signatures, stamps, drawings, logos,
charts) and serves them to the layout sampler so rendered figure/signature regions contain REAL
pixels instead of CSS gradients — the V1/V2 gradient fakes were the reason the detector overfit.
"""
