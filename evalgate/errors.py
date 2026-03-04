class EvalgateError(Exception):
    """Base for every error the gate reports as a failed layer instead of a crash."""


class ConfigError(EvalgateError):
    pass


class DatasetError(EvalgateError):
    pass


class GoldenDriftError(DatasetError):
    """The golden file no longer matches its manifest."""


class CandidateError(EvalgateError):
    pass


class CandidateUnavailable(CandidateError):
    """The candidate needs credentials or an SDK that is not present."""


class JudgeError(EvalgateError):
    pass
