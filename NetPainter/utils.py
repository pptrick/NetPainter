# Software License Agreement (MIT License)
#
# Copyright (C) 2019 Chuanyu Pan (pancy17@mails.tsinghua.edu.cn)
# All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and 
#associated documentation files (the "Software"), to deal in the Software without restriction, 
#including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, 
#and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, 
#subject to the following conditions:
# * Redistributions of source code must retain the above copyright
#   notice, this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright
#   notice, this list of conditions and the following disclaimer in the
#   documentation and/or other materials provided with the distribution.
# * Neither the name of the Tsinghua University nor the
#   names of its contributors may be used to endorse or promote products
#   derived from this software without specific prior written permission.
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES 
#OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE 
#LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR 
#IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


'''utils (important tools)'''

import cairo
import math

def draw_rectangle(cr, bottom_x, bottom_y, length, width, r=0, g=0, b=0):
    '''
    (bottom_x, bottom_y) is the "up left" corner of the rectangle
    length gives increment on x axis
    width gives increment on y axis
    '''
    cr.set_source_rgb(r, g, b)
    cr.rectangle(bottom_x, bottom_y, length, width)
    cr.fill()


def draw_triangle(cr, pt1, pt2, pt3, r, g, b):
    '''
    pt1,pt2,pt3 are the three points of your triangle, as a tuple (x,y)
    '''
    cr.set_source_rgb(r, g, b)
    cr.move_to(pt1[0], pt1[1])
    cr.line_to(pt2[0], pt2[1])
    cr.line_to(pt3[0], pt3[1])
    cr.line_to(pt1[0], pt1[1])
    cr.fill()


def draw_parallelogram(cr, pt1, pt2, pt3, pt4, r=0, g=0, b=0):
    '''
    pt1=(x0,y0),pt2=(x1,y1),pt3=(x2,y2),pt4(x3,y3) is a series of points following the 
    order of drawing a parallelgram, but actually it can be an arbitrary quadrangle 
    '''
    cr.set_source_rgb(r, g, b)
    cr.move_to(pt1[0], pt1[1])
    cr.line_to(pt2[0], pt2[1])
    cr.line_to(pt3[0], pt3[1])
    cr.line_to(pt4[0], pt4[1])
    cr.line_to(pt1[0], pt1[1])
    cr.fill()


def draw_layer_slice(cr, center_x, center_y, length, height, width, r=0, g=0, b=0):
    '''
    (center_x,center_y) is the left-side face center position of the layer on 2D surface
    length is on y=x dir 
    height is on y dir
    width is on x dir
    '''
    #set coordinates
    top_y=center_y-height/2-length/2.828
    middle_y=center_y-height/2+length/2.828
    bottom_y=center_y+height/2+length/2.828
    left_x=center_x-length/2.828+width
    right_x=center_x+length/2.828+width

    #draw front
    draw_rectangle(cr, left_x-width, middle_y, width, height, r*0.8, g*0.8, b*0.8)

    #draw top
    draw_parallelogram(cr, (left_x-width, middle_y), (right_x-width, top_y), (right_x, top_y), (left_x, middle_y), r*1.2, g*1.2, b*1.2)

    #draw side
    draw_parallelogram(cr, (left_x, bottom_y), (left_x, middle_y), (right_x, top_y), (right_x, center_y+height/2-length/2.828), r, g, b)


def draw_kernel_graph(cr, pt1, pt2, pt3, pt4, target, kernel_size_x, kernel_size_y):
    '''
    pt1/pt2/pt3/pt4 are four corner points of the kernel square on a layer, as a tuple (x,y)
    target is the point where the kernel converge, also as a tuple (x,y)
    '''
    cr.set_source_rgba(0, 0, 0, 1)
    cr.set_line_width(2)
    #set four corners and converge point
    x1,y1 = pt1
    x2,y2 = pt2
    x3,y3 = pt3
    x4,y4 = pt4
    x0,y0 = target
    #draw the quadrangle on the layer with solid line 
    cr.set_dash([])
    cr.move_to(x1,y1)
    cr.line_to(x2,y2)
    cr.line_to(x3,y3)
    cr.line_to(x4,y4)
    cr.line_to(x1,y1)
    cr.stroke()
    #draw the connections between four corners and central point with dotted line
    cr.set_dash([4.0])
    cr.move_to(x1, y1)
    cr.line_to(x0, y0)
    cr.stroke()
    cr.move_to(x2, y2)
    cr.line_to(x0, y0)
    cr.stroke()
    cr.move_to(x3, y3)
    cr.line_to(x0, y0)
    cr.stroke()
    cr.move_to(x4, y4)
    cr.line_to(x0, y0)
    cr.stroke()
    #add kernel size notes
    cr.set_source_rgb(0, 0, 0)
    cr.select_font_face('arial', cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_NORMAL)
    cr.set_font_size(25)
    tex_x1=(x1+x2)/2
    tex_y1=(y1+y2)/2-15
    tex_x2=(x4+x1)/2-20
    tex_y2=(y4+y1)/2
    kernel_size_x=str(kernel_size_x)
    kernel_size_y=str(kernel_size_y)
    cr.move_to(tex_x1, tex_y1)
    cr.show_text(kernel_size_x)
    cr.move_to(tex_x2,tex_y2)
    cr.show_text(kernel_size_y)

def draw_encoder_shape(cr, center_x, center_y, length, height, width, r=0, g=0, b=0):
    '''
    An Encoder Drawer
    (center_x,center_y) is the left-side face center position of the Encoder on 2D surface
    width is the total length on x dir
    (height, length) is the side face parameter
    '''
    l_pt1=(center_x-length/2.828, center_y-(0.5*height-length/2.828))
    l_pt2=(center_x+length/2.828, center_y-(0.5*height+length/2.828))
    l_pt3=(center_x+length/2.828, center_y+(0.5*height-length/2.828))
    l_pt4=(center_x-length/2.828, center_y+(0.5*height+length/2.828))
    r_pt1=(center_x+width-length/2.828, center_y-(0.5*height-length/2.828))
    r_pt2=(center_x+width+length/2.828, center_y-(0.5*height+length/2.828))
    r_pt3=(center_x+width+length/2.828, center_y+(0.5*height-length/2.828))
    r_pt4=(center_x+width-length/2.828, center_y+(0.5*height+length/2.828))
    c_pt=(center_x+width/2, center_y)

    if width < length/1.414:
        draw_triangle(cr, l_pt2, l_pt3, c_pt, r*0.8, g*0.8, b*0.8) 
    draw_triangle(cr, l_pt1, l_pt2, c_pt, r*1.2, g*1.2, b*1.2)
    draw_triangle(cr, l_pt1, l_pt4, c_pt, r*0.7, g*0.7, b*0.7)
    if width > height:
        draw_triangle(cr, r_pt1, r_pt2, c_pt, r*1.3, g*1.3, b*1.3)
    else:
        draw_triangle(cr, l_pt4, l_pt3, c_pt, r*0.5, g*0.5, b*0.5)
    draw_triangle(cr, r_pt1, r_pt4, c_pt, r*0.8, g*0.8, b*0.8)
    draw_parallelogram(cr, r_pt1, r_pt2, r_pt3, r_pt4, r, g, b)

    


