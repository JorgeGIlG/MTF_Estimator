# -*- coding: utf-8 -*-
"""
/***************************************************************************
 MtfEstimator
                                 A QGIS plugin
 Robust ESF, PSF, FWHM & MTF estimation from low quality targets and synthetic edge creation.
 Generated by Plugin Builder: http://g-sherman.github.io/Qgis-Plugin-Builder/
                             -------------------
        begin                : 2020-07-12
        copyright            : (C) 2020 by Jorge Gil
        email                : jorge.gil@tutanota.de
        git sha              : $Format:%H$
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/
 This script initializes the plugin, making it known to QGIS.
"""


# noinspection PyPep8Naming
def classFactory(iface):  # pylint: disable=invalid-name
    """Load MtfEstimator class from file MtfEstimator.

    :param iface: A QGIS interface instance.
    :type iface: QgsInterface
    """
    #
    from .mtf_estimator import MtfEstimator
    return MtfEstimator(iface)
