# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
# 修改为更明确的路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# -- Project information -----------------------------------------------------

project = 'KD'
copyright = '2025-2026, Scientific-Artificial-Intelligence-Lab'
author = 'Scientific-Artificial-Intelligence-Lab'

# The full version, including alpha/beta/rc tags
# Keep in sync with ``pyproject.toml``.
release = '1.0.0'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.mathjax',  # 添加数学公式支持
    'sphinx.ext.intersphinx',  # 添加跨文档引用支持
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

# 设置语言支持
language = 'en'
locale_dirs = ['locale/']   # 存放翻译文件的目录
gettext_compact = False     # 禁用 gettext 压缩
languages = ['en', 'zh_CN']  # 支持的语言列表

# 添加源码文档设置
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': True,
    'exclude-members': '__weakref__'
}

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'alabaster'  # 使用默认的 alabaster theme

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# 添加语言切换按钮的配置
html_theme_options = {
    'github_user': 'Scientific-Artificial-Intelligence-Lab',
    'github_repo': 'kd',
    'description': 'Knowledge Discovery Documentation',
    'fixed_sidebar': True,
    'show_powered_by': False,  # 是否显示 "Powered by Sphinx"
    'github_banner': True,     # 添加 GitHub 角标
    'github_button': True,     # 添加 GitHub star/fork 按钮
}
