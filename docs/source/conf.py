# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
sys.path.insert(0, os.path.abspath('../..'))  # Add the project root directory to Python path

# -- Project information -----------------------------------------------------

project = 'kd'
copyright = '2025, Scientific-Artificial-Intelligence-Lab'
author = 'Scientific-Artificial-Intelligence-Lab'

# The full version, including alpha/beta/rc tags
release = '0.0.1'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    # 'sphinx_rtd_theme', 
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
}