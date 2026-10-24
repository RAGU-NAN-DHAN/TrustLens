class TrustLensError(Exception):
    error_code = "INTERNAL_ERROR"


class VerificationError(TrustLensError):
    error_code = "VERIFICATION_FAILED"


class CorrectionError(TrustLensError):
    error_code = "CORRECTION_FAILED"
