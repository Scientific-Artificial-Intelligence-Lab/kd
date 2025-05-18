# equation_renderer.py

import matplotlib.pyplot as plt
import os


def render_latex_to_image(latex_str, output_path = None, font_size=16, dpi=300, background_color='white',
                          base_figsize_width=8, base_figsize_height=2 # 这俩用于动态调整图片大小
                          ):
    try:
        # 动态调整图像宽度：基本宽度 + 每N个字符增加一点宽度 调整系数可能需要根据实际的LaTeX复杂度和字体大小进行微调
        # LaTeX 字符串中的 "$" 和空格等也会计入长度，但作为粗略估计可以接受
        estimated_chars = len(latex_str)
        # 每10个字符增加1英寸宽度，最小宽度为base_figsize_width，最大宽度为30英寸
        adjusted_width = base_figsize_width + (max(0, estimated_chars - 50) / 15.0) 
        adjusted_width = max(base_figsize_width, min(adjusted_width, 40)) 

        fig, ax = plt.subplots(figsize=(adjusted_width, base_figsize_height), facecolor=background_color)
        
        # textprops = {'fontsize': font_size} # 如果需要进一步控制文本属性
        ax.text(0.5, 0.5, latex_str, size=font_size, ha='center', va='center', color="black", wrap=True)
        ax.axis('off') # 关闭坐标轴和边框
        
        if output_path is not None:
            plt.savefig(output_path, dpi=dpi, bbox_inches='tight', pad_inches=0.1, facecolor=background_color)
            print(f"[Equation Renderer INFO] 图像已成功保存到: {output_path}")
        
        plt.show()
        
    except Exception as e:
        print(f"[Equation Renderer Error] 渲染 LaTeX 到图像时发生错误: {e}")
        print(f"  错误的 LaTeX 内容可能是: {latex_str}")
    finally:
        if 'fig' in locals(): # 确保 fig 变量存在
            plt.close(fig) # 关闭图形，释放内存

    pass



