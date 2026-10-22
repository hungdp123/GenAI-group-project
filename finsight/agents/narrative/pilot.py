"""Compatibility entry point for corpus.maintenance.pilot."""
from finsight.agents.narrative.corpus.maintenance import pilot as _implementation

main = _implementation.main


def __getattr__(name: str):
    return getattr(_implementation, name)


if __name__ == "__main__":
    main()
