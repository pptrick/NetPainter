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


'''A network drawer in the style of slices'''

import math
import cairo
import NetPainter.utils as utils

class Model:
    def __init__(self, img_w=1000, img_h=1000, interval=10):
        '''
        Model is the base of your net graph. 

        Here we use modules to store layers information.
        eg. A ReLu, A Softmax, A set of convolution layers are all modules.

        self.lenth is used to estimate the total length of the network graph
        in order to adjust the graph to a proper position.

        interval is used to set the distance between layers.
        '''
        self.img_w=img_w
        self.img_h=img_h

        '''set drawing surface'''
        self.ims = cairo.ImageSurface(cairo.FORMAT_ARGB32, self.img_w, self.img_h)
        self.cr = cairo.Context(self.ims)

        '''set module'''
        self.module_num=0
        self.modules=[]
        self.module_interval=interval
        self.lenth=0 # Total length of the network 
        self.half_lenth=self.lenth/2
        self.use_module={'Softmax':False, 'ReLu':False, 'Conv2d':False, 'BN':False, 'Residual':False, 'Maxpooling':False, 'Encoder':False}

        '''font settings'''
        self.font='arial'
        self.font_size=30
        self.word_trans_x=0
        self.word_trans_y=0

        '''color settings'''
        self.color={'Softmax':(0.8, 0.8, 0.1), 'ReLu':(0.3, 0.6, 1.0), 'Conv2d':(0.9, 0.5, 0.8), 'BN':(0.5, 0.1, 0.8), 'Residual':(0.1, 0.88, 0.88), 'Maxpooling':(0.9, 0.1, 0.1), 'Encoder':(0.3, 0.3, 0.8)}


    def __len__(self):
        return self.lenth
 

    def __getitem__(self, item):
        return self.modules[item]


    def _Auto_Set(self, res_x, res_y, channel, draw_h, draw_l, draw_w):
        '''
        Set draw_h, draw_l, draw_w according to res_x, res_y, channel value.
        '''
        ndraw_h, ndraw_l, ndraw_w = draw_h, draw_l, draw_w
        if draw_w is None:
            ndraw_w = 10 * math.sqrt(channel)
        if draw_h is None and draw_l is None:
            ndraw_l = res_x
            ndraw_h = res_y
        elif draw_h is None:
            ndraw_h = draw_l
        elif draw_l is None:
            ndraw_l = draw_h
        return ndraw_h, ndraw_l, ndraw_w

    def customize(self, name='mylayer', res_x=0, res_y=0, channel=0, slice_num=1, draw_h=None, draw_l=None, draw_w=None, r=0.3, g=0.3, b=0.3, blank=10, notation=True):
        '''
        Design your own net layer/block

        name : Use it to set the layer's name
        res_x/res_y/channel : Your layer's/block's parameters
        slice_num : The number of slice you want to draw
        draw_h/draw_l/draw_w : The height/length/width of the layer you want to draw
        r/g/b : The color of your layer/block
        blank: Distance between this module and the next one.
        notation: If it's set to True, the parameter information will be shown on the top of the layer
        '''
        draw_h, draw_l, draw_w = self._Auto_Set(res_x=res_x, res_y=res_y, channel=channel, draw_h=draw_h, draw_l=draw_l, draw_w=draw_w)
        properti={}
        properti['name']=name
        properti['begin_pos_origin']=self.lenth
        properti['begin_pos']=self.lenth
        properti['res_x']=res_x
        properti['res_y']=res_y
        properti['channel']=channel
        properti['slice_num']=slice_num
        properti['draw_h']=draw_h
        properti['draw_l']=draw_l
        properti['draw_w']=draw_w
        properti['blank']=blank
        properti['notation']=notation
        self.lenth += (draw_w + self.module_interval) * slice_num + blank - self.module_interval
        self.modules.append(properti)
        self.module_num += 1
        self.use_module[name]=True
        self.color[name]=(r, g, b)


    def Conv2d(self, res_x=0, res_y=0, channel=0, kernel=None, slice_num=1, has_ReLu=False, draw_h=None, draw_l=None, draw_w=None, blank=10, notation=True):
        '''
        A 2d convolution layer generator

        res_x/res_y : The actual resolution of the input, which will be shown as a layer parameter(eg.res_x*res_y*channel : 256*256*3) 
        channel: The actual channel number of the input
        kernel: If it is set, a convolution kernel will be shown on the layer; The input can be either an integer or a tuple.
        slice_num: The number of slice you want to draw
        has_Relu: If it's set to True, a ReLu layer will be added to the end of the Conv2d
        draw_h/draw_l/draw_w : The height/length/width of a layer you want to draw
        blank: Distance between this module and the next one.
        notation: If it's set to True, the parameter information will be shown on the top of the layer
        '''
        draw_h, draw_l, draw_w = self._Auto_Set(res_x=res_x, res_y=res_y, channel=channel, draw_h=draw_h, draw_l=draw_l, draw_w=draw_w)
        properti={}
        properti['name']='Conv2d'
        properti['begin_pos_origin']=self.lenth
        properti['begin_pos']=self.lenth
        properti['res_x']=res_x
        properti['res_y']=res_y
        properti['channel']=channel
        properti['kernel']=kernel
        properti['slice_num']=slice_num
        properti['draw_h']=draw_h
        properti['draw_l']=draw_l
        properti['draw_w']=draw_w
        properti['blank']=blank
        properti['notation']=notation
        self.lenth += (draw_w + self.module_interval) * slice_num
        self.modules.append(properti)
        self.module_num += 1
        self.use_module['Conv2d']=True
        if has_ReLu: # add a ReLu layer
            self.ReLu(draw_h=draw_h, draw_l=draw_l, draw_w=draw_w, blank=blank)
        else:
            self.lenth += blank - self.module_interval


    def Residual(self, res_x=0, res_y=0, channel=0, slice_num=1, draw_h=None, draw_l=None, draw_w=None, blank=10, notation=True):
        '''
        A Residual block generator
        '''
        draw_h, draw_l, draw_w = self._Auto_Set(res_x=res_x, res_y=res_y, channel=channel, draw_h=draw_h, draw_l=draw_l, draw_w=draw_w)
        properti={}
        properti['name']='Residual'
        properti['begin_pos_origin']=self.lenth
        properti['begin_pos']=self.lenth
        properti['res_x']=res_x
        properti['res_y']=res_y
        properti['channel']=channel
        properti['slice_num']=slice_num
        properti['draw_h']=draw_h
        properti['draw_l']=draw_l
        properti['draw_w']=draw_w
        properti['blank']=blank
        properti['notation']=notation
        self.lenth += (draw_w + self.module_interval) * slice_num  + blank - self.module_interval
        self.modules.append(properti)
        self.module_num += 1
        self.use_module['Residual']=True


    def Maxpooling(self, res_x=0, res_y=0, channel=0, slice_num=1, draw_h=None, draw_l=None, draw_w=None, blank=10, notation=True):
        '''
        A Maxpooling generator
        '''
        draw_h, draw_l, draw_w = self._Auto_Set(res_x=res_x, res_y=res_y, channel=channel, draw_h=draw_h, draw_l=draw_l, draw_w=draw_w)
        properti={}
        properti['name']='Maxpooling'
        properti['begin_pos_origin']=self.lenth
        properti['begin_pos']=self.lenth
        properti['res_x']=res_x
        properti['res_y']=res_y
        properti['channel']=channel
        properti['slice_num']=slice_num
        properti['draw_h']=draw_h
        properti['draw_l']=draw_l
        properti['draw_w']=draw_w
        properti['blank']=blank
        properti['notation']=notation
        self.lenth += (draw_w + self.module_interval) * slice_num + blank - self.module_interval
        self.modules.append(properti)
        self.module_num += 1
        self.use_module['Maxpooling']=True


    def ReLu(self, res_x=0, res_y=0, channel=0, draw_h=None, draw_l=None, draw_w=None, blank=10):
        '''
        A ReLu generator
        '''
        draw_h, draw_l, draw_w = self._Auto_Set(res_x=res_x, res_y=res_y, channel=channel, draw_h=draw_h, draw_l=draw_l, draw_w=draw_w)
        properti={}
        properti['name']='ReLu'
        properti['begin_pos_origin']=self.lenth
        properti['begin_pos']=self.lenth
        properti['res_x']=res_x
        properti['res_y']=res_y
        properti['channel']=channel
        properti['slice_num']=1 # default set to 1
        properti['draw_h']=draw_h
        properti['draw_l']=draw_l
        properti['draw_w']=draw_w
        properti['blank']=blank
        properti['notation']=False # default set to False
        self.lenth += draw_w + blank
        self.modules.append(properti)
        self.module_num += 1
        self.use_module['ReLu']=True


    def Softmax(self, res_x=0, res_y=0, channel=0, draw_h=None, draw_l=None, draw_w=None, blank=10, notation=True):
        '''
        A Softmax generator
        '''
        draw_h, draw_l, draw_w = self._Auto_Set(res_x=res_x, res_y=res_y, channel=channel, draw_h=draw_h, draw_l=draw_l, draw_w=draw_w)
        properti={}
        properti['name']='Softmax'
        properti['begin_pos_origin']=self.lenth
        properti['begin_pos']=self.lenth
        properti['res_x']=res_x
        properti['res_y']=res_y
        properti['channel']=channel
        properti['slice_num']=1 # default set to 1
        properti['draw_h']=draw_h
        properti['draw_l']=draw_l
        properti['draw_w']=draw_w
        properti['blank']=blank
        properti['notation']=notation
        self.lenth += draw_w + blank
        self.modules.append(properti)
        self.module_num += 1
        self.use_module['Softmax']=True


    def BN(self, res_x=0, res_y=0, channel=0, draw_h=None, draw_l=None, draw_w=None, blank=10):
        '''
        A BatchNorm generator
        '''
        draw_h, draw_l, draw_w = self._Auto_Set(res_x=res_x, res_y=res_y, channel=channel, draw_h=draw_h, draw_l=draw_l, draw_w=draw_w)
        properti={}
        properti['name']='BN'
        properti['begin_pos_origin']=self.lenth
        properti['begin_pos']=self.lenth
        properti['res_x']=res_x
        properti['res_y']=res_y
        properti['channel']=channel
        properti['slice_num']=1 # default set to 1
        properti['draw_h']=draw_h
        properti['draw_l']=draw_l
        properti['draw_w']=draw_w
        properti['blank']=blank
        properti['notation']=False # default set to False
        self.lenth += draw_w + blank
        self.modules.append(properti)
        self.module_num += 1
        self.use_module['BN']=True

    def Encoder(self, draw_h=300, draw_l=300, draw_w=400, blank=10):
        '''
        A Encoder generator
        '''
        properti={}
        properti['name']='Encoder'
        properti['begin_pos_origin']=self.lenth
        properti['begin_pos']=self.lenth
        properti['draw_h']=draw_h
        properti['draw_l']=draw_l
        properti['draw_w']=draw_w
        properti['blank']=blank
        properti['notation']=False # default set to False
        self.lenth += draw_w + blank
        self.modules.append(properti)
        self.module_num += 1
        self.use_module['Encoder']=True


    def _update_lenth(self):
        '''
        change module's begin position and adjust the module to a proper position
        '''
        self.half_lenth = self.lenth / 2
        for module in self.modules:
            module['begin_pos']=module['begin_pos_origin'] - self.half_lenth + self.img_w/2

    
    def _cal_kernel_size(self, module):
        kernel=module['kernel']
        rate=0
        if module['res_x']==0 or module['res_y']==0:
            gamma=0.4
        else:
            if isinstance(kernel, int):
                rate=kernel/max(module['res_x'],module['res_y'])
            elif isinstance(kernel, tuple):
                rate=kernel[0]/max(module['res_x'],module['res_y'])
            gamma=0.1+0.9*math.sqrt(rate)
        return gamma


    def _Draw_Kernel(self, module):
        if(module['name']=='Conv2d') and module['kernel'] is not None:
            gamma=self._cal_kernel_size(module)
            x0,y0=(module['begin_pos']+module['draw_w'], 0.5*self.img_h)
            pt1=(x0-gamma*module['draw_l']/2.828, y0-gamma*(0.5*module['draw_h']-module['draw_l']/2.828))
            pt2=(x0+gamma*module['draw_l']/2.828, y0-gamma*(0.5*module['draw_h']+module['draw_l']/2.828))
            pt3=(x0+gamma*module['draw_l']/2.828, y0+gamma*(0.5*module['draw_h']-module['draw_l']/2.828))
            pt4=(x0-gamma*module['draw_l']/2.828, y0+gamma*(0.5*module['draw_h']+module['draw_l']/2.828))
            target=(x0+module['blank'],y0)
            kernel=module['kernel']
            if isinstance(kernel, int):
                utils.draw_kernel_graph(self.cr, pt1, pt2, pt3, pt4, target, kernel, kernel)
            elif isinstance(kernel, tuple):
                kernel_x, kernel_y=kernel
                utils.draw_kernel_graph(self.cr, pt1, pt2, pt3, pt4, target, kernel_x, kernel_y)


    def _Draw_Graph(self):
        '''Draw layers '''
        self._update_lenth()
        for module in self.modules:
            #draw an encoder
            if module['name']=='Encoder':
                center_x = module['begin_pos']
                center_y = self.img_h / 2
                r, g, b=self.color[module['name']]
                utils.draw_encoder_shape(self.cr, center_x, center_y, length=module['draw_l'], height=module['draw_h'], width=module['draw_w'],
                                        r=r, g=g, b=b)
                continue
            #draw other layers
            for i in range(module['slice_num']):
                center_x = module['begin_pos'] + (module['draw_w'] + self.module_interval) * i
                center_y = self.img_h / 2
                r, g, b=self.color[module['name']]
                utils.draw_layer_slice(self.cr, center_x, center_y, length=module['draw_l'], height=module['draw_h'], width=module['draw_w'],
                                        r=r, g=g, b=b)
                self._Draw_Kernel(module)


    def _write_text(self):
        '''add parameters notation on the graph'''
        self.cr.set_source_rgb(0, 0, 0)
        self.cr.select_font_face(self.font, cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_NORMAL)
        self.cr.set_font_size(self.font_size)

        for module in self.modules:
            if module['notation']==True:
                context=str(module['res_x'])+'*'+str(module['res_y'])+'*'+str(module['channel'])
                pos_x = module['begin_pos']+module['draw_l']/2.828+self.word_trans_x
                pos_y = self.img_h/2-module['draw_h']/2-module['draw_l']/2.8-15-self.word_trans_y
                self.cr.move_to(pos_x, pos_y)
                self.cr.show_text(context)


    def _add_notes(self):
        '''add notes at right-bottom corner'''
        level=0
        for module in self.use_module:
            if self.use_module[module]:
                #draw an encoder
                if module=='Encoder':
                    r, g, b=self.color[module]
                    utils.draw_encoder_shape(self.cr, self.img_w-300, self.img_h-100-level, length=30, height=30, width=50,
                                            r=r, g=g, b=b)
                #draw a slice
                else:
                    r, g, b=self.color[module]
                    utils.draw_layer_slice(self.cr, self.img_w-300, self.img_h-100-level, length=25, height=30, width=50,
                                                r=r, g=g, b=b)
                #add text
                self.cr.set_source_rgb(0, 0, 0)
                self.cr.select_font_face(self.font, cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_NORMAL)
                self.cr.set_font_size(40)
                self.cr.move_to(self.img_w-200, self.img_h-100-level)
                self.cr.show_text(module)
                level+=80
        

    def set_font(self, font='arial', font_size=30, trans_x=0, trans_y=0):
        '''change font settings'''
        self.font=font
        self.font_size=font_size
        self.word_trans_x=trans_x
        self.word_trans_y=trans_y


    def set_color(self, layer_name, r, g, b):
        '''change layer color'''
        self.color[layer_name] = (r, g, b)    
                    

    def Draw(self, output_filename='network.png', add_note=True, add_para=True):
        '''
        Draw your graph after all layers have already set
        [This is necessary]
        '''
        #add notes
        if add_note:
            self._add_notes()

        #add graph
        self._Draw_Graph()

        #add parameters
        if add_para:
            self._write_text()

        #write out
        self.ims.write_to_png(output_filename)




