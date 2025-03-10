@echo off
setlocal

REM This make.bat was generated to compile the documentation with Sphinx.
REM Usage: make.bat [target]
REM Targets:
REM      html    - Build the documentation in HTML.
REM      clean   - Clean the build directory (_build).

REM Define the command to invoke Sphinx using Python.
set SPHINXBUILD=python -m sphinx

REM Source directory and build directory.
set SOURCEDIR=.
set BUILDDIR=_build

if "%1" == "html" goto html
if "%1" == "clean" goto clean

echo Uso: make.bat [target]
echo.
echo "    html    - to generate the documentation in HTML"
echo "    clean   - to clean the _build directory"
goto end

:html
%SPHINXBUILD% -b html %SOURCEDIR% %BUILDDIR%
if errorlevel 1 goto error
echo.
echo "Build completed. The HTML pages are in %BUILDDIR%."
goto end

:clean
if exist %BUILDDIR% rmdir /s /q %BUILDDIR%
echo.
echo "The build directory has been cleaned."
goto end

:error
echo.
echo "An error occurred during the build."
goto end

:end
endlocal 