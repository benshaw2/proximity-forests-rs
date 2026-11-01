@ECHO OFF
set SPHINXBUILD=python -m sphinx
set SOURCEDIR=.
set BUILDDIR=_build

if "%1" == "" goto help
%SPHINXBUILD% -W --keep-going -M %1 %SOURCEDIR% %BUILDDIR%
goto end

:help
%SPHINXBUILD% -M help %SOURCEDIR% %BUILDDIR%

:end
