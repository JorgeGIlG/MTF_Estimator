# -*- coding: utf-8 -*-
"""
/***************************************************************************
 MtfEstimator
                                 A QGIS plugin
 Robust ESF, PSF, FWHM & MTF estimation from low quality targets and synthetic edge creation.
 Generated by Plugin Builder: http://g-sherman.github.io/Qgis-Plugin-Builder/
                              -------------------
        begin                : 2020-07-12
        git sha              : $Format:%H$
        copyright            : (C) 2020 by Jorge Gil
        email                : jorge.gil@tutanota.de
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 3 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/
"""


from qgis.PyQt.QtCore import QSettings, QTranslator, QCoreApplication
from PyQt5.QtWidgets import QAction, QPlainTextEdit
from .mtf_estimator_algorithm import Mtf, Transect, sigmoid
import gdal, ogr, osr
import numpy as np
# Initialize Qt resources from file resources.py
from .resources import *
# Import the code for the dialog
from .mtf_estimator_dialog import MtfEstimatorDialog
import os.path

from qgis.PyQt.QtGui import QIcon
from qgis.PyQt.QtWidgets import QAction
from qgis.core import (
    QgsProject, 
    Qgis, 
    QgsRasterLayer, 
    QgsVectorLayer, 
    QgsMapLayerProxyModel
)


class MtfEstimator:
    """QGIS Plugin Implementation."""

    def __init__(self, iface):
        """Constructor.

        :param iface: An interface instance that will be passed to this class
            which provides the hook by which you can manipulate the QGIS
            application at run time.
        :type iface: QgsInterface
        """
        # Save reference to the QGIS interface
        self.iface = iface
        # initialize plugin directory
        self.plugin_dir = os.path.dirname(__file__)
        # initialize locale
        locale = QSettings().value('locale/userLocale')[0:2]
        locale_path = os.path.join(
            self.plugin_dir,
            'i18n',
            'MtfEstimator_{}.qm'.format(locale))

        if os.path.exists(locale_path):
            self.translator = QTranslator()
            self.translator.load(locale_path)
            QCoreApplication.installTranslator(self.translator)

        # Declare instance attributes
        self.actions = []
        self.menu = self.tr(u'&MTF Estimator')        

        # Check if plugin was started the first time in current QGIS session
        # Must be set in initGui() to survive plugin reloads
        self.first_start = None

    # noinspection PyMethodMayBeStatic
    def tr(self, message):
        """Get the translation for a string using Qt translation API.

        We implement this ourselves since we do not inherit QObject.

        :param message: String for translation.
        :type message: str, QString

        :returns: Translated version of message.
        :rtype: QString
        """
        # noinspection PyTypeChecker,PyArgumentList,PyCallByClass
        return QCoreApplication.translate('MtfEstimator', message)


    def add_action(
        self,
        icon_path,
        text,
        callback,
        enabled_flag=True,
        add_to_menu=True,
        add_to_toolbar=True,
        status_tip=None,
        whats_this=None,
        parent=None):
        """Add a toolbar icon to the toolbar.

        :param icon_path: Path to the icon for this action. Can be a resource
            path (e.g. ':/plugins/foo/bar.png') or a normal file system path.
        :type icon_path: str

        :param text: Text that should be shown in menu items for this action.
        :type text: str

        :param callback: Function to be called when the action is triggered.
        :type callback: function

        :param enabled_flag: A flag indicating if the action should be enabled
            by default. Defaults to True.
        :type enabled_flag: bool

        :param add_to_menu: Flag indicating whether the action should also
            be added to the menu. Defaults to True.
        :type add_to_menu: bool

        :param add_to_toolbar: Flag indicating whether the action should also
            be added to the toolbar. Defaults to True.
        :type add_to_toolbar: bool

        :param status_tip: Optional text to show in a popup when mouse pointer
            hovers over the action.
        :type status_tip: str

        :param parent: Parent widget for the new action. Defaults None.
        :type parent: QWidget

        :param whats_this: Optional text to show in the status bar when the
            mouse pointer hovers over the action.

        :returns: The action that was created. Note that the action is also
            added to self.actions list.
        :rtype: QAction
        """
                        
        icon = QIcon(icon_path)
        action = QAction(icon, text, parent)
        action.triggered.connect(callback)
        action.setEnabled(enabled_flag)

        if status_tip is not None:
            action.setStatusTip(status_tip)

        if whats_this is not None:
            action.setWhatsThis(whats_this)

        if add_to_toolbar:
            # Adds plugin icon to Plugins toolbar
            self.iface.addToolBarIcon(action)

        if add_to_menu:
            self.iface.addPluginToMenu(
                self.menu,
                action)

        self.actions.append(action)

        return action

    def initGui(self):
        """Create the menu entries and toolbar icons inside the QGIS GUI."""

        icon_path = ':/plugins/mtf_estimator/icon.png'
        self.add_action(
            icon_path,
            text=self.tr(u'MTF Estimator'),
            callback=self.run,
            parent=self.iface.mainWindow())

        # will be set False in run()
        self.first_start = True


    def unload(self):
        """Removes the plugin menu item and icon from QGIS GUI."""
        for action in self.actions:
            self.iface.removePluginMenu(
                self.tr(u'&MTF Estimator'),
                action)
            self.iface.removeToolBarIcon(action)

        
    def console(self, *message):
        message = [str(i) for i in message]
        self.dlg.plainTextEdit.appendPlainText(" ".join(message))        
        
        
    def finish(self):        
        self.dlg.done(0)        
        
        
    def run_mtf_algo(self):
        #self.dlg.runButton.setEnabled(False)
        self.console("__START__")
        raster_layer = self.dlg.mMapRasterLayerComboBox.currentLayer()
        band_n = self.dlg.mRasterBandComboBox.currentBand()
                
        gdal_layer = gdal.Open(raster_layer.source(), gdal.GA_ReadOnly)
        gt = list(gdal_layer.GetGeoTransform())
        xsize = gdal_layer.RasterXSize
        ysize = gdal_layer.RasterYSize
        band = gdal_layer.GetRasterBand(band_n)
        raster_srs = osr.SpatialReference()
        raster_srs.ImportFromWkt(gdal_layer.GetProjection())
        vlayer = self.dlg.mMapVectorLayerComboBox.currentLayer()      
        vector_srs = osr.SpatialReference()
        vector_srs.ImportFromWkt(vlayer.crs().toWkt())

        # OJO!!!!
        # https://gdal.org/tutorials/osr_api_tut.html#crs-and-axis-order
        import osgeo
        if int(osgeo.__version__[0]) >= 3:
            # GDAL 3 changes axis order: https://github.com/OSGeo/gdal/issues/1546
            raster_srs.SetAxisMappingStrategy(osgeo.osr.OAMS_TRADITIONAL_GIS_ORDER)
            vector_srs.SetAxisMappingStrategy(osgeo.osr.OAMS_TRADITIONAL_GIS_ORDER)

        if str(raster_srs) is '':
            coord_transform = None
            self.console('WARNING: Raster with no CRS')    
            gt[5] = -1*gt[5]
        else:            
            coord_transform = osr.CoordinateTransformation(vector_srs, raster_srs)

        
        self.console(vector_srs.GetName())
        self.console("")
        self.console(raster_srs.GetName())
        self.console("")
        self.console(coord_transform)
        
        
        
        memlayer_drv = ogr.GetDriverByName('Memory')
        memlayer_ds = memlayer_drv.CreateDataSource('')
        memlayer = memlayer_ds.CreateLayer('aoi', raster_srs, geom_type=ogr.wkbPolygon)
        memlayer.CreateField(ogr.FieldDefn('id', ogr.OFTInteger))
        featureDefn = memlayer.GetLayerDefn()
        
        
        for qgs_feature in vlayer.getFeatures():
            featureDefn = memlayer.GetLayerDefn()
            memfeat = ogr.Feature(featureDefn)
            geom = qgs_feature.geometry()
            self.console(geom.asWkt())
            geom = geom.asWkb()
            geom = ogr.CreateGeometryFromWkb(geom)
            if not coord_transform is None:
                geom.Transform(coord_transform)
            self.console(geom)
            
            memfeat.SetGeometry(geom)
            memlayer.CreateFeature(memfeat)
            

        # Get extent in raster coords
        e = np.array(memlayer.GetExtent()).copy()
        e = np.reshape(e, [2,2])
        e = np.array(np.meshgrid(e[0], e[1]))
        E = e.T.reshape(-1, 2)
        m = np.reshape(np.array(gt).copy(),[2,3])
        A = m[:,0]
        m = m[:,1:]
        M = np.linalg.inv(m)
        col_list, row_list = np.matmul(M,(E-A).T)
        pxoffset = 5
        col_min = np.int(np.max([np.floor(np.min(col_list)) - pxoffset,1]))
        col_max = np.int(np.min([np.ceil(np.max(col_list))+pxoffset, xsize-1]))
        row_min = np.int(np.max([np.floor(np.min(row_list)) - pxoffset,1]))
        row_max = np.int(np.min([np.ceil(np.max(row_list))+pxoffset, ysize-1]))
        sub_gt = gt
        sub_gt[0] = gt[0] + gt[1]*col_min + gt[2]*row_min
        sub_gt[3] = gt[3] + gt[4]*col_min + gt[5]*row_min
        sub_xsize = col_max-col_min
        sub_ysize = row_max-row_min        
        
        memraster_drv = gdal.GetDriverByName('MEM')
        memraster = memraster_drv.Create('', sub_xsize, sub_ysize, 1, band.DataType)
        
        memraster.SetProjection(gdal_layer.GetProjection())
        memraster.SetGeoTransform(sub_gt)
        memband = memraster.GetRasterBand(1)                
        memband.WriteArray(np.zeros([sub_ysize, sub_xsize]))
        gdal.RasterizeLayer(memraster, [1], memlayer, burn_values=[1])
        mask = memband.ReadAsArray(0, 0, sub_xsize, sub_ysize)                
        memband.WriteArray(mask*band.ReadAsArray(col_min, row_min, sub_xsize, sub_ysize))
        mask = None
        
        try:            
            mtf = Mtf(memraster, logfunc=self.console)
        except Exception:
            #self.console(Exception)
            self.console("*** Unable to estimate ***")            
            self.console("__END__")
        else:           
            self.console("__END__")
            
        #self.dlg.runButton.setEnabled(True)    

    def set_band(self):
        self.dlg.mRasterBandComboBox.setLayer(self.dlg.mMapRasterLayerComboBox.currentLayer())
    
    def show_help(self):
        from PyQt5.QtCore import QUrl
        from PyQt5.QtGui import QDesktopServices
        QDesktopServices.openUrl(QUrl('https://github.com/JorgeGIlG/MTF_Estimator'))

    def run(self):
        # Create the dialog with elements (after translation) and keep reference
        # Only create GUI ONCE in callback, so that it will only load when the plugin is started
        if self.first_start == True:
            self.first_start = False
            self.dlg = MtfEstimatorDialog()                                                                         
            self.dlg.closeButton.clicked.connect(self.finish)
            self.dlg.runButton.clicked.connect(self.run_mtf_algo)
            self.dlg.helpButton.clicked.connect(self.show_help)
            self.dlg.mMapRasterLayerComboBox.setFilters(QgsMapLayerProxyModel.RasterLayer)
            self.dlg.mMapRasterLayerComboBox.layerChanged.connect(self.set_band)
            self.set_band()
            self.dlg.mMapVectorLayerComboBox.setFilters(QgsMapLayerProxyModel.VectorLayer)
        # show the dialog
        self.dlg.show()

