"""Legacy preference package stub retained only for internal submodule imports."""


def __getattr__(name: str):
    raise ImportError(
        "src.training.preference is no longer a supported public import surface. "
        "Use src.training.supervision for supervision data/projections and "
        "src.training.judges for public judge backends."
    )
