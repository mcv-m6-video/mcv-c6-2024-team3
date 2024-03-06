import numpy as np
from tqdm import tqdm
import cv2
import os



def box_iou_batch(boxes_a, boxes_b): #Computes the IoU for a multiple bboxes at the same time and returns a matrix with the IoU of each pair
	def box_area(box):
		return box[2] * box[3]

	area_a = box_area(boxes_a.T)
	area_b = box_area(boxes_b.T)

	top_left = np.maximum(boxes_a[:, None, :2], boxes_b[:, :2])
	bottom_right = np.minimum(boxes_a[:, None, :2] + boxes_a[:, None, 2:], boxes_b[:, :2] + boxes_b[:, 2:])
	area_inter = np.prod(np.clip(bottom_right - top_left, a_min=0, a_max=None), 2)
		
	return area_inter / (area_a[:, None] + area_b - area_inter)




def non_max_suppression(predictions, confs, iou_threshold): #Implements NMS and returns a boolean array where False indicates the elements to be removed
	rows, _ = predictions.shape
	
	sort_index = np.flip(confs.argsort())
	boxes = predictions[sort_index]

	# boxes = predictions[:, :4]
	# categories = predictions[:, 5]
	ious = box_iou_batch(boxes, boxes)

	ious = ious - np.eye(rows)
	keep = np.ones(rows, dtype=bool)

	for index, iou in enumerate(ious):
		if not keep[index]:
			continue

		condition = (iou > iou_threshold)
		keep = keep & ~condition

	return keep[sort_index.argsort()]

def remove_overlap(bboxes, confs, thr): #For each frame, filters out those BB with an IoU higher than a threshold by applying NMS
	#Perform NMS
	selected_boxes = non_max_suppression(np.array(bboxes),np.array(confs), thr)

	#Get the positions where the array is False
	false_positions = np.where(np.array(selected_boxes) == False)[0]

	#Remove positions given by the NMS
	bboxes_nms = [val for idx, val in enumerate(bboxes) if idx not in false_positions]
	# print(df)

	return bboxes_nms

class Tracker:
	def __init__(self, iou_threshold=0.5):
		self.iou_threshold = iou_threshold
		self.tracks = []
		self.track_id = 1

	def update(self, detections):
		if not self.tracks:
			# If no tracks, start new tracks for all detections
			for detection in detections:
				self.tracks.append({
					"id": self.track_id,
					"bbox": detection
				})
				self.track_id += 1
		else:
			#Indices to remove those tracks that have no correspondence on the actual frame
			remove_idx = []

			#Get bboxes and compute iou matrix
			bboxes_last = np.array([track['bbox'] for track in self.tracks])
			bboxes_actual = np.array(detections)
			ious = box_iou_batch(bboxes_last, bboxes_actual)

			#Search for matches based on IoU 
			for i, track in enumerate(self.tracks):
				best_iou = 0
				best_detection = None

				for j, detection in enumerate(detections):
					iou = ious[i][j]
					# iou = box_iou_batch(np.array(track["bbox"]).reshape(1,-1), np.array(detection).reshape(1,-1))
					if iou > best_iou:
						best_iou = iou
						best_detection = detection
						idx = j

				if best_iou >= self.iou_threshold: #Match is found
					track["bbox"] = best_detection
					detections.remove(best_detection)
					ious = np.delete(ious, idx, axis=1)
				elif best_iou < self.iou_threshold: #Not found any match
					remove_idx.append(i)

			#Remove those tracks that have no correspondence on the actual frame		
			self.tracks = [val for idx, val in enumerate(self.tracks) if idx not in remove_idx]
					
			#Create new tracks for the unmatched detections
			for detection in detections:
				self.tracks.append({
					"id": self.track_id,
					"bbox": detection,
				})
				self.track_id += 1

				
if __name__ == "__main__":

	predictions_path = '/ghome/group02/C6/Week2/finetuned_preds'
	frames_path = '/ghome/group02/C6/Week1/framesOriginal'
	output_path = '/ghome/group02/C6/Week2/output_finetuned'
	visualize = True
	valid_ids = [0,1,2,3,5,7]

	threshold = 0.4
	thr_NMS = 0.9
	nms_bool = True
	thr = 0.4

	#Variables for the text format
	font = cv2.FONT_HERSHEY_SIMPLEX
	font_scale = 1
	font_thickness = 2
	
	#Remove the overlapping bboxes
	# df = remove_overlap(predictions_path, 0.9)
	#Get the frames list and path
	preds_list = sorted(os.listdir(predictions_path))
	frames_path_sorted = sorted(os.listdir(frames_path))

	#Create tracker object
	tracker = Tracker(iou_threshold=thr)

	#Dictionary to store colors assigned to each ID
	id_colors = {}
	
	with open(f'/ghome/group02/C6/Week2/tracking_results/pretrained/tracking_iou_YOLO_finetuned_{threshold}_{str(nms_bool)}_{thr}.txt', 'w') as file_pred:
		for i, frame in tqdm(enumerate(preds_list)):
			#Load the frame
			img_frame = cv2.imread(f'{frames_path}/{frames_path_sorted[i]}', cv2.IMREAD_COLOR)
			im_height, im_width, _ = img_frame.shape
			bboxes_frame=[]
			confs = []

			with open(os.path.join(predictions_path, frame), 'r') as file:
				for line in file:
					# print(line)
					id, x, y, w, h, conf = line.split()
					if (int(id) in valid_ids) and float(conf)>threshold:
						xtl = int(float(x) * im_width - float(w) * im_width / 2)
						ytl = int(float(y) * im_height - float(h) * im_height / 2)
						w = int(float(w) * im_width)
						h = int(float(h) * im_height)
						bboxes_frame.append([xtl, ytl, w, h])
						confs.append(float(conf))
						
				
			# Obtain the BBoxes
			# detections = np.array(bboxes_frame)
			if nms_bool == True:
				bboxes_frame = remove_overlap(bboxes_frame, confs, thr_NMS)
			#Do the update to find the matches
			tracker.update(bboxes_frame)
			# print(f"Tracks after frame {frame}: {tracker.tracks}")
			
			for track in tracker.tracks:
				id = int(track['id'])
				tlx, tly, w, h = track['bbox']
				tlx, tly, w, y = int(tlx), int(tly), int(w), int(h)
				if visualize == True:
					if id not in id_colors:
						# Assign a new color for the ID
						id_colors[id] = np.random.randint(0, 255, 3)  # Random color
						id_colors[id] = int(id_colors[id][0]), int(id_colors[id][1]), int(id_colors[id][2])
					
					color = id_colors[id]

					text_org = (tlx, tly - 10) #Place the text right above the bbox

					cv2.rectangle(img_frame, (tlx, tly), (tlx+w, tly+h), color, 2)
					cv2.putText(img_frame, str(id), text_org, font, font_scale, color, font_thickness)
				file_pred.write(f"{i+1},{id},{tlx},{tly},{w},{h},{1},{1},{1},{1}\n")

			if visualize == True:
				cv2.imwrite(f'{output_path}/{frames_path_sorted[i]}', img_frame)
