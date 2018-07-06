'''
blah
'''
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
from moviepy.editor import VideoFileClip
import math
import os
import timeit
import pickle


# Global parameters

# Gaussian smoothing
kernel_size = 3

# Canny Edge Detector
low_threshold = 100
high_threshold = 300

# Region-of-interest vertices
# We want a trapezoid shape, with bottom edge at the bottom of the image
trap_bottom_width = 1  # width of bottom edge of trapezoid, expressed as percentage of image width
trap_top_width = 0.15  # ditto for top edge of trapezoid
trap_bottom_height = 0
trap_height = 0.5  # height of the trapezoid expressed as percentage of image height

# Hough Transform
rho = 2 # distance resolution in pixels of the Hough grid
theta = np.pi/180 # angular resolution in radians of the Hough grid
threshold = 15	 # minimum number of votes (intersections in Hough grid cell)
min_line_length = 15 #minimum number of pixels making up a line
max_line_gap = 10	# maximum gap in pixels between connectable line segments


# Helper functions
def grayscale(img):
	"""Applies the Grayscale transform
	This will return an image with only one color channel
	but NOTE: to see the returned image as grayscale
	you should call plt.imshow(gray, cmap='gray')"""
	return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
def abs_sobel_thresh(img, orient='x', thresh_min=0, thresh_max=255):
    # Apply x or y gradient with the OpenCV Sobel() function
    # and take the absolute value
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3))
    # Rescale back to 8 bit integer
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    # Create a copy and apply the threshold
    binary_output = np.zeros_like(scaled_sobel)
    # Here I'm using inclusive (>=, <=) thresholds, but exclusive is ok too
    binary_output[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1

    # Return the result
    return binary_output

def mag_thresh(img, thresh_min=0, thresh_max=255):
    # Take both Sobel x and y gradients
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=9)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=9)
    # Calculate the gradient magnitude
    gradmag = np.sqrt(sobelx**2 + sobely**2)
    # Rescale to 8 bit
    scale_factor = np.max(gradmag)/255 
    gradmag = (gradmag/scale_factor).astype(np.uint8) 
    # Create a binary image of ones where threshold is met, zeros otherwise
    binary_output = np.zeros_like(gradmag)
    binary_output[(gradmag >= thresh_min) & (gradmag <= thresh_max)] = 1

    # Return the binary image
    return binary_output

	
def canny(img, low_threshold, high_threshold):
	#"""Applies the Canny transform"""
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    l_channel = hls[:,:,1]
    s_channel = hls[:,:,2]
    
    # Apply sobel in x direction on L and S channel
    l_channel_sobel_x = abs_sobel_thresh(l_channel,'x', 20, 200)
    s_channel_sobel_x = abs_sobel_thresh(s_channel,'x', 20, 200)
    sobel_combined_x = cv2.bitwise_or(s_channel_sobel_x, l_channel_sobel_x)
    
    # Apply magnitude sobel
    l_channel_mag = mag_thresh(l_channel, 20, 200)
    s_channel_mag = mag_thresh(s_channel, 20, 200)
    mag_combined = cv2.bitwise_or(l_channel_mag, s_channel_mag)
    
    # Combine all the sobel filters
    mask_combined = cv2.bitwise_or(mag_combined, sobel_combined_x)
    #return cv2.Canny(img, low_threshold, high_threshold)
    return mask_combined
     

def gaussian_blur(img, kernel_size):
	"""Applies a Gaussian Noise kernel"""
	return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
	"""
	Applies an image mask.
	
	Only keeps the region of the image defined by the polygon
	formed from `vertices`. The rest of the image is set to black.
	"""
	#defining a blank mask to start with
	global car
	mask = np.zeros_like(img)   
	
	#defining a 3 channel or 1 channel color to fill the mask with depending on the input image
	if len(img.shape) > 2:
		channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
		ignore_mask_color = (255,) * channel_count
	else:
		ignore_mask_color = 255
		
	#filling pixels inside the polygon defined by "vertices" with the fill color	
	print(car)
	#if car == False or it<1 or light ==False or reset == False:
	if car == False or it<1 or light ==False or reset == False:
		cv2.fillPoly(mask, vertices, ignore_mask_color)
	else:
		vertices_left = np.array([[\
		((img.shape[0]-left_b_frames[it-1])/left_m_frames[it-1]-25, img.shape[0])\
       ,((0.5*img.shape[0]-left_b_frames[it-1])/left_m_frames[it-1]-3, img.shape[0]*0.5)\
       ,((0.5*img.shape[0]-left_b_frames[it-1])/left_m_frames[it-1]+3, img.shape[0]*0.5)\
       ,((img.shape[0]-left_b_frames[it-1])/left_m_frames[it-1]+25, img.shape[0])]]\
		, dtype=np.int32)		
		vertices_right = np.array([[\
		((img.shape[0]-right_b_frames[it-1])/right_m_frames[it-1]-25, img.shape[0]),\
       ((0.5*img.shape[0]-right_b_frames[it-1])/right_m_frames[it-1]-3, img.shape[0]*0.5),\
       ((0.5*img.shape[0]-right_b_frames[it-1])/right_m_frames[it-1]+3, img.shape[0]*0.5),\
       ((img.shape[0]-right_b_frames[it-1])/right_m_frames[it-1]+25, img.shape[0])]]\
		, dtype=np.int32)	
		print(vertices_right)  

		#filling pixels inside the polygon defined by "vertices" with the fill color	
		cv2.fillPoly(mask, vertices_left, ignore_mask_color)
		cv2.fillPoly(mask, vertices_right, ignore_mask_color)	
		#returning the image only where mask pixels are nonzero
	masked_image = cv2.bitwise_and(img, mask)
	return masked_image

def draw_lines(img, lines, color_left=[0, 255, 0], color_right=[255, 0, 0], thickness=10):

	global it
	global car
	global reset
	global right_b_frames
	global right_m_frames
	global left_b_frames
	global left_m_frames
	"""
	NOTE: this is the function you might want to use as a starting point once you want to 
	average/extrapolate the line segments you detect to map out the full
	extent of the lane (going from the result shown in raw-lines-example.mp4
	to that shown in P1_example.mp4).  
	
	Think about things like separating line segments by their 
	slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
	line vs. the right line.  Then, you can average the position of each of 
	the lines and extrapolate to the top and bottom of the lane.
	
	This function draws `lines` with `color` and `thickness`.	
	Lines are drawn on the image inplace (mutates the image).
	If you want to make the lines semi-transparent, think about combining
	this function with the weighted_img() function below
	"""
	# In case of error, don't draw the line(s)
	car = False
	y = img.shape[0]
	if lines is None:
		return
	if len(lines) == 0:
		return
	draw_right = True
	draw_left = True
	
	# Find slopes of all lines
	# But only care about lines where abs(slope) > slope_threshold
	slope_threshold = 0.3
	slopes = []
	new_lines = []

	for line in lines:
		x1, y1, x2, y2 = line[0]  # line = [[x1, y1, x2, y2]]
		length = math.sqrt((x1-x2)**2+(y1-y2)**2)
		# Calculate slope
		if x2 - x1 == 0.:  # corner case, avoiding division by 0
			slope = 999.  # practically infinite slope
		else:
			slope = (y2 - y1) / (x2 - x1)
			
		# Filter lines based on slope
		if abs(slope) > slope_threshold and abs(slope) < 4 and  abs(slope)!=999:
			slopes.append(slope)
			new_lines.append(line)
		if slope >40 and length > 100:
			car = True
			print('slpoe:%f length:%f'%(slope,length) )   
     
#		if 0< abs(slope) <0.1 and length>200 and length<800:
#			print('slpoe:%f length:%f'%(slope,length) )   
#			car = True    
   
	lines = new_lines
	
	# Split lines into right_lines and left_lines, representing the right and left lane lines
	# Right/left lane lines must have positive/negative slope, and be on the right/left half of the image
	right_lines = []
	left_lines = []
	for i, line in enumerate(lines):
		x1, y1, x2, y2 = line[0]
		img_x_center = img.shape[1] / 2  # x coordinate of center of image
		if slopes[i] > 0 and x1 > img_x_center and x2 > img_x_center:
			right_lines.append(line)
		elif slopes[i] < 0 and x1 < img_x_center and x2 < img_x_center:
			left_lines.append(line)
			
	# Run linear regression to find best fit line for right and left lane lines
	# Right lane lines
	right_lines_x = []
	right_lines_y = []
	right_weights = []
	right_slopes = []

	for line in right_lines:
		x1, y1, x2, y2 = line[0]
		slope = (y2-y1)/(x2-x1)
		intercept = y1 - slope*x1
		length = np.sqrt((y2-y1)**2+(x2-x1)**2)
		
		right_weights.append(length**2)
		right_slopes.append((slope,intercept))
				
		right_lines_x.append(x1)
		right_lines_x.append(x2)
		
		right_lines_y.append(y1)
		right_lines_y.append(y2)
		
	if len(right_lines_x) > 0:
		right_m, right_b = np.polyfit(right_lines_x, right_lines_y, 1)  # y = m*x + b
		right_m, right_b = np.dot(right_weights,  right_slopes) /np.sum(right_weights)
	else:
		#right_m, right_b = 1, 1
  
		right_m = right_m_frames[it-1]
		right_b = right_b_frames[it-1]
		#draw_right = False 
		
	# Left lane lines
	left_lines_x = []
	left_lines_y = []
	left_weights = []
	left_slopes = []
	
	for line in left_lines:
		x1, y1, x2, y2 = line[0]
		slope = (y2-y1)/(x2-x1)
		intercept = y1 - slope*x1
		length = np.sqrt((y2-y1)**2+(x2-x1)**2)

		left_weights.append(length**2)
		left_slopes.append((slope,intercept))
		
		left_lines_x.append(x1)
		left_lines_x.append(x2)
		
		left_lines_y.append(y1)
		left_lines_y.append(y2)
		
	if len(left_lines_x) > 0:
		left_m, left_b = np.polyfit(left_lines_x, left_lines_y, 1)  # y = m*x + b
		left_m, left_b = np.dot(left_weights, left_slopes) /np.sum(left_weights)
	else:
		#left_m, left_b = 1, 1
		left_m = left_m_frames[it-1]
		left_b = left_b_frames[it-1]
		#draw_left = False
	if it >0 and reset == False: 
		if abs( (y-right_b)/right_m - (y-right_b_frames[it-1])/right_m_frames[it-1] )  >30 \
			or abs(right_m-right_m_frames[it-1])>0.5 :     
			right_m = right_m_frames[it-1]
			right_b = right_b_frames[it-1]   
			reset = True
		if abs( (y-left_b)/left_m - (y-left_b_frames[it-1])/left_m_frames[it-1] )  >30 \
			or abs(left_m-left_m_frames[it-1])>0.5 :
			left_m = left_m_frames[it-1]
			left_b = left_b_frames[it-1]
			reset = True 
	else: 
		reset = False
 
    
#	
	# Find 2 end points for right and left lines, used for drawing the line
	# y = m*x + b --> x = (y - b)/m
 
 ####################---------average--------------###########################

	if it <= 1:
		right_b_,right_m_,left_b_,left_m_ = right_b,right_m,left_b,left_m
	else:
#		if abs(right_m - right_m_frames[it-1])>0.4:
#			right_m = right_m_frames[it-1]+np.sign(right_m - right_m_frames[it-1])*0.2
#		if abs(left_m - left_m_frames[it-1])>0.4:
#			left_m = left_m_frames[it-1]+np.sign(left_m - left_m_frames[it-1])*0.2

		right_b_ = (right_b + right_b_frames[it-2] +right_b_frames[it-1])/3
		right_m_ = (right_m + right_m_frames[it-2] +right_m_frames[it-1])/3
		left_b_ = (left_b + left_b_frames[it-2] +left_b_frames[it-1])/3
		left_m_ = (left_m + left_m_frames[it-2] +left_m_frames[it-1])/3

	right_b_frames.append(right_b_)
	right_m_frames.append(right_m_)
	left_b_frames.append(left_b_)
	left_m_frames.append(left_m_)
	it = it + 1	
	# Convert calculated end points from float to int
	y1 = img.shape[0]
	y2 = img.shape[0] * (1 - trap_height)
	
	y_coord = np.linspace(450,650,9) 
	x_coord_left = (y_coord - left_b) / left_m
	x_coord_right = (y_coord - right_b) / right_m
	y_coord = y_coord.astype(int)
	x_coord_left = x_coord_left.astype(int)
	x_coord_right = x_coord_right.astype(int)
 
#####################-----draw lines------################################ 
#	right_x1 = (y1 - right_b) / right_m
#	right_x2 = (y2 - right_b) / right_m
#	
#	left_x1 = (y1 - left_b) / left_m
#	left_x2 = (y2 - left_b) / left_m
#	
#	# Convert calculated end points from float to int
#	y1 = int(y1)
#	y2 = int(y2)
#	right_x1 = int(right_x1)
#	right_x2 = int(right_x2)
#	left_x1 = int(left_x1)
#	left_x2 = int(left_x2)
	color_right = (0,255,0)
	color_left = (0,255,0)

	# Draw the right and left lines on image
	if draw_right:
		for x,y in zip(x_coord_right,y_coord):
			for line in right_lines:
				x1, y1, x2, y2 = line[0]
				if y1<y<y2 or y2<y<y1:
					color_right = (255,0,0)
			cv2.circle(img,(x,y), 5, color_right, -1)
		#cv2.line(img, (right_x1, y1), (right_x2, y2), color_right, thickness)
	if draw_left:
		for x,y in zip(x_coord_left,y_coord):
			for line in left_lines:
				x1, y1, x2, y2 = line[0]
				if y1<y<y2 or y2<y<y1:
					color_left = (255,0,0)
			cv2.circle(img,(x,y), 5, color_left, -1)
   #cv2.line(img, (left_x1, y1), (left_x2, y2), color_left, thickness)
	
def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
	"""
	`img` should be the output of a Canny transform.
		
	Returns an image with hough lines drawn.
	"""
	lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
	line_img = np.zeros((*img.shape, 3), dtype=np.uint8)  # 3-channel RGB image
	draw_lines(line_img, lines)
	return lines,line_img

# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
	"""
	`img` is the output of the hough_lines(), An image with lines drawn on it.
	Should be a blank image (all black) with lines drawn on it.
	
	`initial_img` should be the image before any processing.
	
	The result image is computed as follows:
	
	initial_img * α + img * β + λ
	NOTE: initial_img and img must be the same shape!
	"""
	return cv2.addWeighted(initial_img, α, img, β, λ)

def filter_colors(image):
	"""
	Filter the image to include only yellow and white pixels
	"""
	# Filter white pixels
	white_threshold = 120 #130
	lower_white = np.array([white_threshold, white_threshold, white_threshold])
	upper_white = np.array([255, 255, 255])
	white_mask = cv2.inRange(image, lower_white, upper_white)
	white_image = cv2.bitwise_and(image, image, mask=white_mask)

	# Filter yellow pixels
	hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
	lower_yellow = np.array([90,100,100])
	upper_yellow = np.array([110,255,255])
	yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
	yellow_image = cv2.bitwise_and(image, image, mask=yellow_mask)

	# Combine the two above images
	image2 = cv2.addWeighted(white_image, 1., yellow_image, 1., 0.)

	return image2

def annotate_image_array(image_in):
    
	global light
	start = timeit.default_timer()
	""" Given an image Numpy array, return the annotated image as a Numpy array """
	# Only keep white and yellow pixels in the image, all other pixels become black
	image = filter_colors(image_in)
	
	# Read in and grayscale the image
	gray = grayscale(image)

	# Apply Gaussian smoothing
	blur_gray = gaussian_blur(gray, kernel_size)
#	if blur_gray.mean() < 5:
#		car = True
#	else:
#		car = False
	if light<60 or light>100 :
		light = True
	else:
		light = False
	print(blur_gray.mean())

	# Apply Canny Edge Detector
	edges = canny(image_in, low_threshold, high_threshold)
	#edges = canny(image_in, low_threshold, high_threshold)

	# Create masked edges using trapezoid-shaped region-of-interest
	imshape = image_in.shape
	vertices = np.array([[\
		((imshape[1] * (1 - trap_bottom_width)) // 2, imshape[0]- imshape[0] * trap_bottom_height),\
		((imshape[1] * (1 - trap_top_width)) // 2, imshape[0] - imshape[0] * trap_height),\
		(imshape[1] - (imshape[1] * (1 - trap_top_width)) // 2, imshape[0] - imshape[0] * trap_height),\
		(imshape[1] - (imshape[1] * (1 - trap_bottom_width)) // 2, imshape[0]-imshape[0] * trap_bottom_height)]]\
		, dtype=np.int32)
	masked_edges = region_of_interest(edges, vertices)

	# Run Hough on edge detected image
	lines,line_image = hough_lines(masked_edges, rho, theta, threshold, min_line_length, max_line_gap)
	
	# Draw lane lines on the original image
	initial_image = image_in.astype('uint8')
	annotated_image = weighted_img(line_image, initial_image)
	stop = timeit.default_timer()
	#print (stop - start)
	
	return annotated_image

def annotate_image(input_file, output_file):
	""" Given input_file image, save annotated image to output_file """
	lines,image,gray,blur_gray,edges,masked_edges,initial_image,annotated_image = annotate_image_array(mpimg.imread(input_file))
	plt.imsave(output_file, annotated_image)
	return lines,image,gray,blur_gray,edges,masked_edges,initial_image,annotated_image

def annotate_video(input_file, output_file):
	""" Given input_file video, save annotated video to output_file """
	video = VideoFileClip(input_file)
	annotated_video = video.fl_image(annotate_image_array)
	annotated_video.write_videofile(output_file, audio=False)
	with open('save.obj', 'wb') as fp:
		pickle.dump([right_b_frames,right_m_frames,left_b_frames,left_m_frames], fp)
 

# End helper functions


# Main script
if __name__ == '__main__':
	from optparse import OptionParser
     
	# Configure command line options
	parser = OptionParser()
	parser.add_option("-i", "--input_file", dest="input_file",
					help="Input video/image file")
	parser.add_option("-o", "--output_file", dest="output_file",
					help="Output (destination) video/image file")
	parser.add_option("-I", "--image_only",
					action="store_true", dest="image_only", default=False,
					help="Annotate image (defaults to annotating video)")

	# Get and parse command line options
	options, args = parser.parse_args()

	input_file = options.input_file
	output_file = options.output_file
	image_only = options.image_only
 
	it = 0 
	car = False
	light = False
	reset = False 
	right_b_frames = []
	right_m_frames = []
	left_b_frames = []
	left_m_frames = []
 
	if image_only:
		annotate_image(input_file, output_file)
	else:
		annotate_video(input_file, output_file)
