# Notes on building documentation from python notebooks and python source files

* to build docs
  * use a virtual environment with sphinx, myst-parser, and nbsphinx installed, then run
    * `cd src/masskit_ai/docs`
    * `sphinx-apidoc -o . ../`
    * `sphinx-build -b html . _build`
    * create and use an orphan branch to hold the pages.  First time initialization of this branch is something like
      * 
        ```
        mkdir masskit_ai_nist-pages
        cd masskit_ai_nist-pages
        git init
        git remote add origin <url>
        git checkout --orphan nist-pages

        # copy over the pages

        git add .
        git commit -m "init"
        git push -u origin nist-pages
        ```

* notes
  * source files won't be parsed unless they are in a directory containing __init__.py
  * methods for creating documentation:
    * Each function and class can be documented using restructuredtext placed in the source file.  
      This documentation will automatically appear in the module documentation.
      * examples can be found in our source code and [here](https://thomas-cokelaer.info/tutorials/sphinx/docstring_python.html),
        with the source code that generated the documentation at the bottom of the page. 
    * jupyter notebooks 
      * the names of the notebooks should be added explicitly by putting their filename without extension into the file
        docs/index.rst underneath "toctree".
       * don't include more than one markdown header per markdown cell.  That is, don't put "## header 1" followed by "## header 2" in the same cell.
    * markdown files
      * the names of the markdown files should be added explicitly by putting their filename without extension into the file docs/index.rst underneath "toctree".
    * restructuredtext (*.rst) files, which are widely used and the same format used to document functions and classes.
  * autodoc has problems importing some modules.  If these are modules not to be documented, one can mock up the import using the `autodoc_mock_imports` setting in conf.py
  * because sphinx uses directories that start with underscore, this confuses jekyll.  To turn off jekyll, use the `sphinx.ext.githubpages` extension in conf.py

  