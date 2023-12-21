
BACKGROUND_CATEGORY = ['ground','land','grass','tree','building','wall','sky','lake','water','river','sea','railway','railroad','keyboard','helmet',
                        'cloud','house','mountain','ocean','road','rock','street','valley','bridge','sign',
                        ]

class_names = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
                   'bus', 'car', 'cat', 'chair', 'cow',
                   'diningtable', 'dog', 'horse', 'motorbike', 'person',
                   'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor',
                   ]
                   
new_class_names = ['aeroplane', 'bicycle', 'bird avian', 'boat', 'bottle',
                   'bus', 'car', 'cat', 'chair seat', 'cow',
                   'diningtable', 'dog', 'horse', 'motorbike', 'person with clothes,people,human',
                   'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor screen',
                   ]


class_names_coco = ['person','bicycle','car','motorbike','aeroplane',
                    'bus','train','truck','boat','traffic light',
                    'fire hydrant','stop sign','parking meter','bench','bird',
                    'cat','dog','horse','sheep','cow',
                    'elephant','bear','zebra','giraffe','backpack',
                    'umbrella','handbag','tie','suitcase','frisbee',
                    'skis','snowboard','sports ball','kite','baseball bat',
                    'baseball glove','skateboard','surfboard','tennis racket','bottle',
                    'wine glass','cup','fork','knife','spoon',
                    'bowl','banana','apple','sandwich','orange',
                    'broccoli','carrot','hot dog','pizza','donut',
                    'cake','chair','sofa','pottedplant','bed',
                    'diningtable','toilet','tvmonitor','laptop','mouse',
                    'remote','keyboard','cell phone','microwave','oven',
                    'toaster','sink','refrigerator','book','clock',
                    'vase','scissors','teddy bear','hair drier','toothbrush',
]

new_class_names_coco = ['person with clothes,people,human','bicycle','car','motorbike','aeroplane',
                    'bus','train','truck','boat','traffic light',
                    'fire hydrant','stop sign','parking meter','bench','bird avian',
                    'cat','dog','horse','sheep','cow',
                    'elephant','bear','zebra','giraffe','backpack,bag',
                    'umbrella,parasol','handbag,purse','necktie','suitcase','frisbee',
                    'skis','sknowboard','sports ball','kite','baseball bat',
                    'glove','skateboard','surfboard','tennis racket','bottle',
                    'wine glass','cup','fork','knife','dessertspoon',
                    'bowl','banana','apple','sandwich','orange',
                    'broccoli','carrot','hot dog','pizza','donut',
                    'cake','chair seat','sofa','pottedplant','bed',
                    'diningtable','toilet','tvmonitor screen','laptop','mouse',
                    'remote control','keyboard','cell phone','microwave','oven',
                    'toaster','sink','refrigerator','book','clock',
                    'vase','scissors','teddy bear','hairdrier,blowdrier','toothbrush',
                    ]


BACKGROUND_CATEGORY_COCO = ['ground','land','grass','tree','building','wall','sky','lake','water','river','sea','railway','railroad','helmet',
                        'cloud','house','mountain','ocean','road','rock','street','valley','bridge',
                        ]


class_names_coco_stuff182_dict = {
0: 'unlabeled',
1: 'person with clothes,people,human',#'person',
2: 'bicycle',
3: 'car',
4: 'motorcycle',
5: 'aeroplane',#'airplane',
6: 'bus',
7: 'train',
8: 'truck',
9: 'boat',
10: 'traffic light',
11: 'fire hydrant',
#12: 'street sign',
13: 'stop sign',
14: 'parking meter',
15: 'bench',
16: 'bird avian',#'bird',
17: 'cat',
18: 'dog',
19: 'horse',
20: 'sheep',
21: 'cow',
22: 'elephant',
23: 'bear',
24: 'zebra',
25: 'giraffe',
#26: 'hat',
27: 'backpack,bag',#'backpack',
28: 'umbrella,parasol',#'umbrella',
#29: 'shoe',
#30: 'eye glasses',
31: 'handbag,purse',#'handbag',
32: 'necktie',#'tie',
33: 'suitcase',
34: 'frisbee',
35: 'skis',
36: 'snowboard',
37: 'sports ball',
38: 'kite',
39: 'baseball bat',
40: 'glove',#'baseball glove',
41: 'skateboard',
42: 'surfboard',
43: 'tennis racket',
44: 'bottle',
#45: 'plate',
46: 'wine glass',
47: 'cup',
48: 'fork',
49: 'knife',
50: 'dessertspoon',#'spoon',
51: 'bowl',
52: 'banana',
53: 'apple',
54: 'sandwich',
55: 'orange',
56: 'broccoli',
57: 'carrot',
58: 'hot dog',
59: 'pizza',
60: 'donut',
61: 'cake',
62: 'chair seat',
63: 'couch',
64: 'pottedplant',
65: 'bed',
#66: 'mirror',
67: 'diningtable',
#68: 'window',
#69: 'desk',
70: 'toilet',
#71: 'door',
72: 'tvmonitor screen',#'tv',
73: 'laptop',
74: 'mouse',
75: 'remote control',#'remote',
76: 'keyboard',
77: 'cell phone',
78: 'microwave',
79: 'oven',
80: 'toaster',
81: 'sink',
82: 'refrigerator',
#83: 'blender',
84: 'book',
85: 'clock',
86: 'vase',
87: 'scissors',
88: 'teddy bear',
89: 'hairdrier,blowdrier',#'hair drier',
90: 'toothbrush',
#91: 'hair brush',
92: 'banner',
93: 'blanket',
94: 'branch',
95: 'bridge',
96: 'building-other',
97: 'bush',
98: 'cabinet',
99: 'cage',
100: 'cardboard',
101: 'carpet',
102: 'ceiling-other',
103: 'ceiling-tile',
104: 'cloth',
105: 'clothes',
106: 'clouds',
107: 'counter',
108: 'cupboard',
109: 'curtain',
110: 'desk-stuff',
111: 'dirt',
112: 'door-stuff',
113: 'fence',
114: 'floor-marble',
115: 'floor-other',
116: 'floor-stone',
117: 'floor-tile',
118: 'floor-wood',
119: 'flower',
120: 'fog',
121: 'food-other',
122: 'fruit',
123: 'furniture-other',
124: 'grass',
125: 'gravel',
126: 'ground-other',
127: 'hill',
128: 'house',
129: 'leaves',
130: 'light',
131: 'mat',
132: 'metal',
133: 'mirror-stuff',
134: 'moss',
135: 'mountain',
136: 'mud',
137: 'napkin',
138: 'net',
139: 'paper',
140: 'pavement',
141: 'pillow',
142: 'plant-other',
143: 'plastic',
144: 'platform',
145: 'playingfield',
146: 'railing',
147: 'railroad',
148: 'river',
149: 'road',
150: 'rock',
151: 'roof',
152: 'rug',
153: 'salad',
154: 'sand',
155: 'sea',
156: 'shelf',
157: 'sky-other',
158: 'skyscraper',
159: 'snow',
160: 'solid-other',
161: 'stairs',
162: 'stone',
163: 'straw',
164: 'structural-other',
165: 'table',
166: 'tent',
167: 'textile-other',
168: 'towel',
169: 'tree',
170: 'vegetable',
171: 'wall-brick',
172: 'wall-concrete',
173: 'wall-other',
174: 'wall-panel',
175: 'wall-stone',
176: 'wall-tile',
177: 'wall-wood',
178: 'water-other',
179: 'waterdrops',
180: 'window-blind',
181: 'window-other',
182: 'wood',
}

coco_stuff_categories = [
  "electronic",  # 0
  "appliance",  # 1
  "food things",  # 2, food-things, i.e., cake, donut, pizza, hot dog, carrot, broccoli, orange, sandwich, apple, and banana
  "furniture things",  # 3, furniture-thing
  "indoor",  # 4
  "kitchen",  # 5
  "accessory",  # 6
  "animal",  # 7
  "outdoor",  # 8
  "person",  # 9
  "sports",  # 10
  "vehicle",  # 11

  "ceiling",  # 12
  "floor",  # 13
  "food stuff",  # 14, food-stuff, i.e., food-other, vegetable, salad, and fruit
  "furniture stuff",  # 15
  "raw material",  # 16
  "textile",  # 17
  "wall",  # 18
  "window",  # 19
  "building",  # 20
  "ground",  # 21
  "plant",  # 22
  "sky",  # 23
  "solid",  # 24
  "structural",  # 25
  "water"  # 26
]

coco_stuff_182_to_27 = {
    0: 9, 1: 11, 2: 11, 3: 11, 4: 11, 5: 11, 6: 11, 7: 11, 8: 11, 9: 8, 10: 8, 11: 8, 12: 8,
    13: 8, 14: 8, 15: 7, 16: 7, 17: 7, 18: 7, 19: 7, 20: 7, 21: 7, 22: 7, 23: 7, 24: 7,
    25: 6, 26: 6, 27: 6, 28: 6, 29: 6, 30: 6, 31: 6, 32: 6, 33: 10, 34: 10, 35: 10, 36: 10,
    37: 10, 38: 10, 39: 10, 40: 10, 41: 10, 42: 10, 43: 5, 44: 5, 45: 5, 46: 5, 47: 5, 48: 5,
    49: 5, 50: 5, 51: 2, 52: 2, 53: 2, 54: 2, 55: 2, 56: 2, 57: 2, 58: 2, 59: 2, 60: 2,
    61: 3, 62: 3, 63: 3, 64: 3, 65: 3, 66: 3, 67: 3, 68: 3, 69: 3, 70: 3, 71: 0, 72: 0,
    73: 0, 74: 0, 75: 0, 76: 0, 77: 1, 78: 1, 79: 1, 80: 1, 81: 1, 82: 1, 83: 4, 84: 4,
    85: 4, 86: 4, 87: 4, 88: 4, 89: 4, 90: 4, 91: 17, 92: 17, 93: 22, 94: 20, 95: 20, 96: 22,
    97: 15, 98: 25, 99: 16, 100: 13, 101: 12, 102: 12, 103: 17, 104: 17, 105: 23, 106: 15,
    107: 15, 108: 17, 109: 15, 110: 21, 111: 15, 112: 25, 113: 13, 114: 13, 115: 13, 116: 13,
    117: 13, 118: 22, 119: 26, 120: 14, 121: 14, 122: 15, 123: 22, 124: 21, 125: 21, 126: 24,
    127: 20, 128: 22, 129: 15, 130: 17, 131: 16, 132: 15, 133: 22, 134: 24, 135: 21, 136: 17,
    137: 25, 138: 16, 139: 21, 140: 17, 141: 22, 142: 16, 143: 21, 144: 21, 145: 25, 146: 21,
    147: 26, 148: 21, 149: 24, 150: 20, 151: 17, 152: 14, 153: 21, 154: 26, 155: 15, 156: 23,
    157: 20, 158: 21, 159: 24, 160: 15, 161: 24, 162: 22, 163: 25, 164: 15, 165: 20, 166: 17,
    167: 17, 168: 22, 169: 14, 170: 18, 171: 18, 172: 18, 173: 18, 174: 18, 175: 18, 176: 18,
    177: 26, 178: 26, 179: 19, 180: 19, 181: 24
}

coco_stuff_182_to_171 = {}
cnt = 0
for label_id in coco_stuff_182_to_27:
    if label_id + 1 in [12, 26, 29, 30, 45, 66, 68, 69, 71, 83, 91]:  # note that +1 is added
        continue
    coco_stuff_182_to_171[label_id] = cnt
    cnt += 1

coco_stuff_171_to_27 = {}
cnt = 0
for fine, coarse in coco_stuff_182_to_27.items():
    if fine + 1 in [12, 26, 29, 30, 45, 66, 68, 69, 71, 83, 91]:  # note that +1 is added
        continue
    coco_stuff_171_to_27[cnt] = coarse
    cnt += 1

coco_stuff_27_to_182 = {}
for i in range(27):
    coco_stuff_27_to_182[i] = []
for fine, coarse in coco_stuff_182_to_27.items():
    #if fine + 1 in [12, 26, 29, 30, 45, 66, 68, 69, 71, 83, 91]:  # note that +1 is added
    #    continue
    coco_stuff_27_to_182[coarse].append(fine)