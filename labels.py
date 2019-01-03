#!/usr/bin/python
#
# DS20k labels
#

from collections import namedtuple

#--------------------------------------------------------------------------------
# A list of all labels
#--------------------------------------------------------------------------------

Label = namedtuple('Label',['name','Id','color'])

labels = [

    ###      name               Id   color
    Label(  'BACKGROUND'    ,   0 , (  0,  0,  0)   ),
    Label(  'PERSON'        ,   1 , (255,  0,121)   ),
    Label(  'CAR'           ,   2 , (255, 15, 15)   ),
    Label(  'TRUCK'         ,   3 , (254, 83,  1)   ),
    Label(  'DRIVABLE'      ,   4 , (  0,255,  0)   ),
    Label(  'NONDRIVABLE'   ,   5 , (255,255,  0)   ),
    Label(  'BLOCKER'       ,   6 , (192,192,192)   ),
    Label(  'INFO'          ,   7 , (  0,  0,255)   ),
    Label(  'SKY'           ,   8 , (128,255,255)   ),
    Label(  'BUILDINGS'     ,   9 , ( 83,  0,  0)   ),
    Label(  'NATURE'        ,  10 , (  0, 80,  0)   ),
    Label(  'SLOWAREA'      ,  11 , (  0,255,255)   ),
    Label(  'LINES'         ,  12 , (128,  0,255)   ),

]

#--------------------------------------------------------------------------------
# Create dictionaries for a fast lookup
#--------------------------------------------------------------------------------

name2label = {label.name : label for label in labels}