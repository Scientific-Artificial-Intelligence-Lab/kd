# equation_renderer.py

import matplotlib.pyplot as plt
from .dlga_eq2latex import dlga_eq2latex
from .discover_eq2latex import discover_program_to_latex


def _ensure_math_mode(latex_str: str) -> str:
    if latex_str is None:
        return '$$'
    stripped = latex_str.strip()
    if stripped.startswith('$') and stripped.endswith('$') and len(stripped) >= 2:
        return latex_str
    return f"${latex_str}$"


def render_latex_to_image(
    latex_str,
    output_path=None,
    font_size=16,
    dpi=300,
    background_color='white',
    base_figsize_width=8,
    base_figsize_height=2,
    *,
    show=True,
):
    try:
        latex_str = _ensure_math_mode(str(latex_str))
        # Heuristically adjust figure width based on LaTeX string length.
        estimated_chars = len(latex_str)
        adjusted_width = base_figsize_width + (max(0, estimated_chars - 50) / 15.0) 
        adjusted_width = max(base_figsize_width, min(adjusted_width, 40)) 

        fig, ax = plt.subplots(figsize=(adjusted_width, base_figsize_height), facecolor=background_color)
        
        # Use mathtext to render LaTeX; do not enable wrap to avoid fallback to plain text.
        ax.text(0.5, 0.5, latex_str, size=font_size, ha='center', va='center', color="black")
        ax.axis('off')
        
        if output_path is not None:
            plt.savefig(output_path, dpi=dpi, bbox_inches='tight', pad_inches=0.1, facecolor=background_color)
            print(f"[Equation Renderer INFO] Image successfully saved to: {output_path}")
        
        if show:
            plt.show()
        
    except Exception as e:
        print(f"[Equation Renderer ERROR] Failed to render LaTeX to image: {e}")
        print(f"  Offending LaTeX content: {latex_str}")
    finally:
        if 'fig' in locals():
            try:
                plt.close(fig)
            except TypeError:
                # 在测试环境中, plt.subplots 可能被 monkeypatch 成返回非 Figure 对象
                # 此时直接忽略关闭错误, 以避免干扰调用方。
                plt.close()
