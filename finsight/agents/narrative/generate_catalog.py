"""Compatibility entry point for corpus.maintenance.generate_catalog."""
from finsight.agents.narrative.corpus.maintenance import generate_catalog as _implementation

main = _implementation.main


def __getattr__(name: str):
    return getattr(_implementation, name)


if __name__ == "__main__":
    main()
