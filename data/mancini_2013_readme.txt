J/A+A/562/A126      Light curves of WASP-80 transit events   (Mancini+, 2014)
================================================================================
Physical properties and transmission spectrum of the WASP-80 planetary system
from multi-colour photometry.
    Mancini L., Southworth J., Ciceri S., Dominik M., Henning T.,
    Jorgensen U.G., Lanza A. F., Rabus M., Snodgrass C., Vilela C.,
    Alsubai K.A., Bozza V., Bramich D.M., Calchi Novati S., D'Ago G.,
    Figuera Jaimes R., Galianni P., Gu S.-H., Harpose K., Hinse T.,
    Hundertmark M., Juncher D., Kains N., Korhonen H., Popovas A.,
    Rahvar S., Skottfelt J., Street R., Surdej J., Tsapras Y.,
    Wang X.-B., Wertz O.
   <Astron. Astrophys. 562, A126 (2014)>
   =2014A&A...562A.126M
================================================================================
ADC_Keywords: Stars, double and multiple ; Planets ; Photometry
Keywords: planetary systems - stars: fundamental parameters -
          stars: individual: WASP-80 - techniques: photometric

Abstract:
    WASP-80 is one of only two systems known to contain a hot Jupiter
    which transits its M-dwarf host star. We present eight light curves of
    one transit event, obtained simultaneously using two defocussed
    telescopes. These data were taken through the Bessell I, Sloan
    g'r'i'z' and near-infrared JHK passbands. We use our data to search
    for opacity-induced changes in the planetary radius, but find that all
    values agree with each other. Our data are therefore consistent with a
    flat transmission spectrum to within the observational uncertainties.
    We also measure an activity index of the host star of
    logR'_HK=-4.495, meaning that WASP-80A shows strong chromospheric
    activity. The non-detection of starspots implies that, if they exist,
    they must be small and symmetrically distributed on the stellar
    surface. We model all available optical transit light curves and
    obtain improved physical properties and orbital ephemerides for the
    system.

Description:
    8 light curves of one transit of the extrasolar planetary system
    WASP-80, observed on 2013 June 16, are presented.

    Seven of the datasets were obtained using the MPG/ESO 2.2-m telescope,
    GROND camera and filters similar to Sloan g', r', i', z', and J, H, K
    at the ESO Observatory in La Silla (Chile).

    Another data set was obtained using the 1.54-m Danish telescope, DFOSC
    camera and Bessel-i filter at La Silla (Chile).

Objects:
    ----------------------------------------------------------------
       RA   (2000)   DE       Designation(s)
    ----------------------------------------------------------------
    20 12 40.18  -02 08 39.2  WASP-80 = 2MASS J20124017-0208391
    ----------------------------------------------------------------

File Summary:
--------------------------------------------------------------------------------
 FileName   Lrecl  Records   Explanations
--------------------------------------------------------------------------------
ReadMe         80        .   This file
eso_g.dat      39      162  *Photometry of WASP-80 on 2013/06/16 (g ESO 2.2m)
eso_r.dat      39      156  *Photometry of WASP-80 on 2013/06/16 (r ESO 2.2m)
eso_i.dat      39      156  *Photometry of WASP-80 on 2013/06/16 (i ESO 2.2m)
eso_z.dat      39      157  *Photometry of WASP-80 on 2013/06/16 (z ESO 2.2m)
eso_j.dat      39      508  *Photometry of WASP-80 on 2013/06/16 (J ESO 2.2m)
eso_h.dat      39      508  *Photometry of WASP-80 on 2013/06/16 (H ESO 2.2m)
eso_k.dat      39      508  *Photometry of WASP-80 on 2013/06/16 (K ESO 2.2m)
dk_i.dat       39      200  *Photometry of WASP-80 on 2013/06/16 (I DK 1.54m)
--------------------------------------------------------------------------------
Note on *.dat: Correspond to full table2 of the paper.
--------------------------------------------------------------------------------

See also:
   J/A+A/558/A55    : Trans. planetary system HATS-2 (Mohler-Fischer+, 2013)
   J/A+A/554/A28    : Transiting planetary system Qatar-1 (Covino+, 2013)
   J/A+A/549/A10    : Transiting planetary system GJ1214 (Harpsoe+, 2013)
   J/A+A/551/A11    : Transiting planetary system HAT-P-8 (Mancini+, 2013)
   J/MNRAS/420/2580 : Transiting planetary system HAT-P-13 (Southworth+, 2012)
   J/MNRAS/422/3099 : Transiting planetary system HAT-P-5 (Southworth+, 2012)
   J/MNRAS/408/1680 : Transiting planetary system WASP-2 (Southworth+, 2010)
   J/MNRAS/399/287  : Transiting planetary system WASP-4 (Southworth+, 2009)
   J/MNRAS/396/1023 : Transiting planetary system WASP-5 (Southworth+, 2009)
   J/A+A/527/A8     : Transiting planetary system WASP-7 (Southworth+, 2011)
   J/MNRAS/434/1300 : WASP-15 and WASP-16 light curves (Southworth+, 2013)
   J/MNRAS/426/1338 : Transiting planetary system WASP-17 (Southworth+, 2012)
   J/ApJ/707/167    : Transiting planetary system WASP-18 (Southworth+, 2009)
   J/MNRAS/436/2    : Light curves of WASP-19 transit events (Mancini+, 2013)
   J/A+A/557/A30    : Trans. planetary systems WASP-21 HAT-P-16 (Ciceri+, 2013)
   J/MNRAS/430/2932 : Transiting planetary system WASP-44 (Mancini+, 2013)

Byte-by-byte Description of file: eso_?.dat dk_i.dat
--------------------------------------------------------------------------------
   Bytes Format Units   Label     Explanations
--------------------------------------------------------------------------------
       1  A1    ---     Band      [grizIJHK] Observed band
   3- 15  F13.7 d       BJD       Barycentric JD for the midpoint of observation
                                   (BJD-2400000)
  18- 27  F10.7 mag     mag       Differential magnitude of WASP-80 in Band
  31- 39  F9.7  mag   e_mag       Measurement error of the magnitude
--------------------------------------------------------------------------------

Acknowledgements:
    Luigi Mancini, mancini(at)mpia.de,
     Max Planck Institute for Astronomy, Germany

================================================================================
(End)      Luigi Mancini [Max Planck Inst.], Patricia Vannier [CDS]  24-Jan-2014
