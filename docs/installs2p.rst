Installation of suite2p on Windows
====================================

Install an Anaconda distribution of Python -- Choose Python 3.x and your operating system.

.. highlight:: rst

.. role:: python(code)
    :language: python


1. Download the :python:`environment.yml` file from the suite2p repository (https://github.com/MouseLand/suite2p). You can do this by cloning the repository, or copy-pasting the text from the file into a text document on your local computer.

2. Open an anaconda prompt

3. If you have an older suite2p environment you can remove it with :python:`conda env remove -n suite2p` before creating a new one.

.. image:: _images/pre.png

4. Replace :python:`{PATH_TO_THE_FILE}` with path of :python:`environment.yml` and run :python:`conda env create -f {PATH_TO_THE_FILE}\environment.yml`. In this case, :python:`{PATH_TO_THE_FILE}` is :python:`D:\NAPE_2pBenchmark`

.. image:: _images/path.png

.. image:: _images/env.png

5. To activate this new environment, run :python:`conda activate suite2p`

.. image:: _images/run.png

6. You should see (suite2p) on the left side of the terminal line. Now run :python:`python -m suite2p` and you're all set.

.. image:: _images/final.png