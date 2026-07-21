"""
FinScope — a Bloomberg-style terminal built on FinceptTerminal's data fleet.

A runnable, dependency-light Python TUI that unifies Fincept's no-key market
connectors (crypto, on-chain, macro) behind one `DataHub`, layers a numpy +
Wolfram analytics engine on top, and renders a `rich` live dashboard driven by
Bloomberg-style mnemonic commands (CRYPTO, TOP, DES, GP, CORR, VOL, MACRO, API).

Run it:   python -m finscope
Help:     python -m finscope --help
"""

__version__ = "0.1.0"
__all__ = ["__version__"]
