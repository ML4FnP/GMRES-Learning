#!/usr/bin/env python
# -*- coding: utf-8 -*-



def set_defaults(args, defaults):

    new_args = dict()

    for key in args:
        if args[key] == None:
            if key in defaults.keys():
                new_args[key] = defaults[key]
        else:
            new_args[key] = args[key]

    return new_args
