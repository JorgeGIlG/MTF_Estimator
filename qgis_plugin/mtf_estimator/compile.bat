@echo off
call "C:\Program Files\QGIS 3.12\bin\o4w_env.bat"
call "C:\Program Files\QGIS 3.12\bin\qt5_env.bat"
call "C:\Program Files\QGIS 3.12\bin\py3_env.bat"

@echo on
pyrcc5 -o resources.py resources.qrc



