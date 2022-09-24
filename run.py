#!/usr/bin/env python
# coding: utf-8

import argparse
import cv2 as cv
import numpy as np

import features
import save_figures
import outputs

global descriptor
global descriptors1
global descriptors2
global detector
global keypoints1
global keypoints2

def main():
    # Message from usage
    message = '''run.py [-h]

                --detector     {SIFT, SURF, KAZE, ORB, BRISK,AKAZE}
                --descriptor   {SIFT, SURF, KAZE, BRIEF, ORB, BRISK, AKAZE, FREAK}
                --matcher      {BF, FLANN}'''

    # Create the parser
    parser = argparse.ArgumentParser(description = 'Computer Vision Algorithms',
                                    usage = message)

    # Argument --detector
    parser.add_argument('--detector',
                        action = 'store',
                        choices = ['SIFT', 'SURF', 'KAZE', 'ORB', 'BRISK', 'AKAZE'],
                        required = True,
                        metavar = '',
                        dest = 'detector',
                        help = 'select the detector to be used in this experiment')

    # Argument --descriptor
    parser.add_argument('--descriptor',
                        action = 'store',
                        choices = ['SIFT', 'SURF', 'KAZE', 'BRIEF', 'ORB', 'BRISK', 'AKAZE', 'FREAK'],
                        required = True,
                        metavar = '',
                        dest = 'descriptor',
                        help = 'select the descriptor to be used in this experiment')

    # Argument --matcher
    parser.add_argument('--matcher',
                        action = 'store',
                        choices = ['BF', 'FLANN'],
                        required = True,
                        metavar = '',
                        dest = 'matcher',
                        help = 'select the matcher to be used in this experiment')

    # Execute the parse_args() method
    arguments = parser.parse_args()

    # Initiate Detector and Descriptor

    # Initiate detector selected
    if arguments.detector == 'SIFT':
        detector = features.SIFT()
    elif arguments.detector == 'SURF':
        detector = features.SURF()
    elif arguments.detector == 'KAZE':
        detector = features.SIFT()
    elif arguments.detector == 'ORB':
        detector = features.ORB()
    elif arguments.detector == 'BRISK':
        detector = features.BRISK()
    elif arguments.detector == 'AKAZE':
        detector = features.AKAZE()

    # Initiate descriptor selected
    if arguments.descriptor == 'SIFT':
        descriptor = features.SIFT()
    elif arguments.descriptor == 'SURF':
        descriptor = features.SURF()
    elif arguments.descriptor == 'KAZE':
        descriptor = features.SIFT()
    elif arguments.descriptor == 'BRIEF':
        descriptor = features.BRIEF()
    elif arguments.descriptor == 'ORB':
        descriptor = features.ORB()
    elif arguments.descriptor == 'BRISK':
        descriptor = features.BRISK()
    elif arguments.descriptor == 'AKAZE':
        descriptor = features.AKAZE()
    elif arguments.descriptor == 'FREAK':
        descriptor = features.FREAK()

    # Open and Convert the input image from RGB to GRAYSCALE
    image1 = cv.imread(filename = 'Figures/image1.jpg',
                    flags = cv.IMREAD_GRAYSCALE)

    # Open and Convert the training-set image from RGB to GRAYSCALE
    image2 = cv.imread(filename = 'Figures/image2.jpg',
                    flags = cv.IMREAD_GRAYSCALE)

    # Could not open or find the images
    if image1 is None or image2 is None:
        print('\nCould not open or find the images.')
        exit(0)

    # Find the keypoints and compute
    # the descriptors for input image
    keypoints1, descriptors1 = features.features(image1, detector, descriptor)

    print('\nInput image:\n')

    # Print infos for input image 
    features.prints(keypoints = keypoints1, used_detector=detector, used_descriptor=descriptor, descriptor = descriptors1)

    # Find the keypoints and compute
    # the descriptors for training-set image
    keypoints2, descriptors2 = features.features(image2, detector, descriptor)

    # Print
    print('Training-set image:\n')

    # Print infos for training-set image
    features.prints(keypoints = keypoints2, used_detector=detector, used_descriptor=descriptor, descriptor = descriptors2)

    # Matcher 
    output = features.matcher(image1 = image1,
                            image2 = image2,
                            keypoints1 = keypoints1,
                            keypoints2 = keypoints2,
                            descriptors1 = descriptors1,
                            descriptors2 = descriptors2,
                            matcher = arguments.matcher,
                            descriptor = arguments.descriptor)

    # Save Figure Matcher
    save_figures.saveMatcher(output = output,
                            matcher = arguments.matcher,
                            descriptor = arguments.descriptor)

    # Save keypoints and descriptors into a file
    # from input image
    outputs.saveKeypointsAndDescriptors(keypoints = keypoints1,
                                        descriptors = descriptors1,
                                        matcher = arguments.matcher,
                                        descriptor = arguments.descriptor,
                                        flags = 1)

    # Save keypoints and descriptors into a file
    # from training-set image
    outputs.saveKeypointsAndDescriptors(keypoints = keypoints2,
                                        descriptors = descriptors2,
                                        matcher = arguments.matcher,
                                        descriptor = arguments.descriptor,
                                        flags = 2)


if __name__ == "__main__":
    main()