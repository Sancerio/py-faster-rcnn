#!/usr/bin/env python
import os
import random
import sys
import matplotlib.cm as cmx
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import matplotlib.legend as lgd
import matplotlib.markers as mks
import numpy as np

def get_log_parsing_script():
    return './parselog.sh'

def get_log_file_suffix():
    return '.log'

def get_chart_type_description_separator():
    return '  vs. '

def is_x_axis_field(field):
    x_axis_fields = ['Iters', 'Seconds']
    return field in x_axis_fields

def create_field_index():
    train_key = 'train'
    test_key = 'test'
    field_index = {train_key:{'Iters':0, 'Seconds':1, train_key + ' loss':2,
                              train_key + ' learning rate':3, train_key + ' bbox_loss':4, 
                              train_key + ' cls_loss':5, train_key + ' rpn_loss_bbox':6,
                              train_key + ' rpn_cls_loss':7},
                   test_key:{'Iters':0, 'Seconds':1, test_key + ' accuracy':2,
                             test_key + ' loss':3}}
    fields = set()
    for data_file_type in field_index.keys():
        fields = fields.union(set(field_index[data_file_type].keys()))
    fields = list(fields)
    fields.sort()
    return field_index, fields

def get_supported_chart_types():
    field_index, fields = create_field_index()
    num_fields = len(fields)
    supported_chart_types = []
    for i in xrange(num_fields):
        if not is_x_axis_field(fields[i]):
            for j in xrange(num_fields):
                if i != j and is_x_axis_field(fields[j]):
                    supported_chart_types.append('%s%s%s' % (
                        fields[i], get_chart_type_description_separator(),
                        fields[j]))
    return supported_chart_types

def get_chart_type_description(chart_type):
    supported_chart_types = get_supported_chart_types()
    chart_type_description = supported_chart_types[chart_type]
    return chart_type_description

def get_data_file_type(chart_type):
    description = get_chart_type_description(chart_type)
    data_file_type = description.split()[0]
    return data_file_type

def get_data_file(chart_type, path_to_log):
    return path_to_log + '.' + get_data_file_type(chart_type).lower()

def get_field_descriptions(chart_type):
    description = get_chart_type_description(chart_type).split(
        get_chart_type_description_separator())
    y_axis_field = description[0]
    x_axis_field = description[1]
    return x_axis_field, y_axis_field    

def get_field_indecies(x_axis_field, y_axis_field):    
    data_file_type = get_data_file_type(chart_type)
    fields = create_field_index()[0][data_file_type]
    return fields[x_axis_field], fields[y_axis_field]

def load_data(data_file, field_idx0, field_idx1):
    data = [[], []]
    with open(data_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line[0] != '#':
                fields = line.split()
                try:
                    data0 = float(fields[field_idx0].strip())
                    data1 = float(fields[field_idx1].strip())
                    data[0].append(data0)
                    data[1].append(data1)
                except:
                    print 'do nothing'
    return data

def random_marker():
    markers = mks.MarkerStyle.markers
    num = len(markers.values())
    idx = random.randint(0, num - 1)
    return markers.values()[idx]

def get_data_label(path_to_log):
    label = path_to_log[path_to_log.rfind('/')+1 : path_to_log.rfind(
        get_log_file_suffix())]
    return label

def get_legend_loc(chart_type):
    x_axis, y_axis = get_field_descriptions(chart_type)
    loc = 'lower right'
    if y_axis.find('accuracy') != -1:
        pass
    if y_axis.find('loss') != -1 or y_axis.find('learning rate') != -1:
        loc = 'upper right'
    return loc

def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0)) 
    return (cumsum[N:] - cumsum[:-N]) / N 

def plot_chart(chart_type, path_to_png, path_to_log_list, axarr, pos):
    for path_to_log in path_to_log_list:
        os.system('%s %s' % (get_log_parsing_script(), path_to_log))
        data_file = get_data_file(chart_type, path_to_log)
        x_axis_field, y_axis_field = get_field_descriptions(chart_type)
        x, y = get_field_indecies(x_axis_field, y_axis_field)
        data = load_data(data_file, x, y)

        color = [0, 1, 0]
        label = get_data_label(path_to_log)
        linewidth = 0.75
        ## If there too many datapoints, do not use marker.
##        use_marker = False
        p = np.polyfit(data[0],data[1],2)
        z = np.poly1d(p)
        #print np.shape(smooth)
        #print np.shape(data[1])
        use_marker = False
        if not use_marker:
            print len(data[0])
            print len(data[1])
            axarr[pos].plot(data[0], data[1], label = label, color = color,
                     linewidth = linewidth)

            # Different color for smoothed data
            print 'Current Loss: {}'.format(z(data[0])[-1])
            print 'Min Loss: {}'.format(min(z(data[0])))
            color = [1, 0, 0]
            axarr[pos].plot(data[0], z(data[0]), label = 'fit', color = color,
                     linewidth = linewidth)
        else:
            ok = False
            ## Some markers throw ValueError: Unrecognized marker style
            while not ok:
                try:
                    marker = random_marker()
                    axarr[pos].plot(data[0], data[1], label = label, color = color,
                             marker = marker, linewidth = linewidth)
                    ok = True
                except:
                    pass
    legend_loc = get_legend_loc(chart_type)
    #axarr[pos].set_legend(loc = legend_loc, ncol = 1) # ajust ncol to fit the space
    #axarr[pos].set_title(get_chart_type_description(chart_type))
    #axarr[pos].set_xlabel(x_axis_field)
    axarr[pos].set_ylabel(y_axis_field)  
    #axarr[pos].savefig(path_to_png)     
    #axarr[pos].show()

def print_help():
    print """This script mainly serves as the basis of your customizations.
Customization is a must.
You can copy, paste, edit them in whatever way you want.
Be warned that the fields in the training log may change in the future.
You had better check the data files and change the mapping from field name to
 field index in create_field_index before designing your own plots.
Usage:
    ./plot_log.sh chart_type[0-%s] /where/to/save.png /path/to/first.log ...
Notes:
    1. Supporting multiple logs.
    2. Log file name must end with the lower-cased "%s".
Supported chart types:""" % (len(get_supported_chart_types()) - 1,
                             get_log_file_suffix())
    supported_chart_types = get_supported_chart_types()
    num = len(supported_chart_types)
    for i in xrange(num):
        print '    %d: %s' % (i, supported_chart_types[i])
    exit

def is_valid_chart_type(chart_type):
    return chart_type >= 0 and chart_type < len(get_supported_chart_types())
  
if __name__ == '__main__':
    if len(sys.argv) < 4:
        print_help()
    else:
        chart_type = int(sys.argv[1])
        if not is_valid_chart_type(chart_type):
            print_help()
        path_to_png = sys.argv[2]
        if not path_to_png.endswith('.png'):
            print 'Path must ends with png' % path_to_png
            exit            
        path_to_logs = sys.argv[3:]
        for path_to_log in path_to_logs:
            if not os.path.exists(path_to_log):
                print 'Path does not exist: %s' % path_to_log
                exit
            if not path_to_log.endswith(get_log_file_suffix()):
                print_help()
        ## plot_chart accpets multiple path_to_logs
        chart_types = [4, 6, 12, 14, 10]
        f, axarr = plt.subplots(5)
        for idx, chart_type in enumerate(chart_types):
            plot_chart(chart_type, path_to_png, path_to_logs, axarr, idx)
        plt.subplots_adjust(hspace = .001)
        plt.savefig(path_to_png)
        plt.show()

