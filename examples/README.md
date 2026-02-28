# KD Examples

## Quick Start

```bash
pip install -e .           # from project root
python examples/sga_example.py
```

## Examples

| File | Method | Description |
|------|--------|-------------|
| `sga_example.py` | KD_SGA | SGA discovery: built-in data, N-D, custom data |
| `dlga_example.py` | KD_DLGA | DLGA discovery with full visualization |
| `dscv_example.py` | KD_DSCV | DISCOVER algorithm (regular/FD mode) |
| `dscv_spr_example.py` | KD_DSCV_SPR | DISCOVER + PINN (sparse data) |
| `pysr_example.py` | KD_PySR | PySR symbolic regression (requires pysr) |

## File Structure

Each example follows the same pattern:

1. Data loading (built-in or custom)
2. Model initialization (all parameters documented, defaults commented out)
3. Training
4. Visualization via `kd.viz`
5. (Optional) N-D spatial data section

## Internal Files

| File | Purpose |
|------|---------|
| `_nd_data.py` | Shared N-D synthetic data factory |
| `_gallery_gen.py` | README gallery image generation (developer use) |
