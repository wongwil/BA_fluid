import PySimpleGUI as sg
import numpy as np
import os.path
import cv2
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from predict_instability import load_model, predict_instability
from load_data import normalize_frames, set_labels, get_label_names
    
MODEL_PATH = 'res_net_50_128d_256d_no_dropout_leaky_relu_5x5f.h5'
LABELS = ['still', 'wave', 'nearDrops', 'smallDrops', 'drops', 'foam']
# Frontent-Keys, they have to be unique!
FILE_BROWSE_KEY = "-FILEBROWSE-"
SLIDER_KEY = "-ENHANCE-SLIDER-"
IMAGE_KEY = "-IMAGE-"
LABEL_RESULTS_KEY = "-LABEL-RESULTS-"
LOAD_VIDEO_KEY = "-BUTTON-LOAD-VIDEO-"
CLEAR_KEY = "-CLEAR-"
ANALYZE_KEY = "-ANALYZE-"
CONFIDENCE_OUTPUT_KEY = "-CONFIDENCE-"
SAVE_SESSION_PATH_KEY = "-SAVE-SESSION-PATH-"
SESSION_BROWSE_KEY = "-SESSION-BROWSE-"
LOAD_SESSION_KEY = "-LOAD-SESSION-"
DISTRIBUTION_CANVAS_KEY = "-DISTRIBUTION-CANVAS-KEY-"

def main():
    # First the window layout in 2 columns
    
    file_list_column = [
        [
            sg.Text("Video"),
            sg.In(size=(25, 1), enable_events=True, key=FILE_BROWSE_KEY),
            sg.FileBrowse(file_types=(("Videos", "*.mkv"),("Videos", "*.avi")))
        ],
        [
            sg.Button("Load Video", key=LOAD_VIDEO_KEY), 
            sg.Button("Clear", key=CLEAR_KEY), 
            sg.Button("Analyze", key=ANALYZE_KEY),
            sg.InputText(visible=False, enable_events=True, key=SAVE_SESSION_PATH_KEY),
            sg.FileSaveAs(
                key=SAVE_SESSION_PATH_KEY,
                file_types=(('CSV', '.csv'),))
        ],
        [   
            sg.Text("Session"),
            sg.In(size=(25, 1), enable_events=True, key=SESSION_BROWSE_KEY),
            sg.FileBrowse(file_types=(("CSV-Files", "*.csv"),)),
            sg.Button("Load Session", key=LOAD_SESSION_KEY)
        ],
        [sg.Image(key=IMAGE_KEY)],
        [sg.Slider(range=(1, 1000),
            default_value=1,
            orientation="h",
            size=(80, 15),
            enable_events=True,
            key=SLIDER_KEY)],
        [sg.Canvas(key=DISTRIBUTION_CANVAS_KEY)]
    ]
    
    # For now will only show the name of the file that was chosen
    image_viewer_column = [
        [sg.Text("Result")],
        [sg.Listbox(values=[], 
                    enable_events=True,
                    key=LABEL_RESULTS_KEY, 
                    size=(25, 20))],
        [sg.Text(size=(25,7), key=CONFIDENCE_OUTPUT_KEY)]
    ]
    
    # ----- Full layout -----
    layout = [
        [
            sg.Column(file_list_column),
            sg.VSeperator(),
            sg.Column(image_viewer_column),
        ]
    ]
    
    window = sg.Window("Fluid-Analyzer", layout)
    
    # Model config
    load_model(MODEL_PATH)
    set_labels({'still', 'wave', 'nearDrops', 'smallDrops', 'drops', 'foam'})
    label_names = get_label_names()
    
    # Variables 
    video_loaded = False
    video_analyzed = False
    annotated_frame_labels = []
    y_values = []
    y_confidence = []
    all_frames = []
    file_path = ""
   
    figure_canvas_agg = None
    fig = None

    # Run the Event Loop
    while True:            
        event, values = window.read()
        if event == "Exit" or event == sg.WIN_CLOSED:
            break

        if event == FILE_BROWSE_KEY:
            selection = values[FILE_BROWSE_KEY]
            if selection != "":
                # TODO: save path
                file_path = selection
                print(selection)
        elif event == LOAD_VIDEO_KEY and file_path != "" and video_loaded == False:
            all_frames = read_frames(file_path)
            first_frame = all_frames[0]
            window[IMAGE_KEY].update(data=get_image_bytes(first_frame))
            video_loaded = True
            
            # update slider length
            window.FindElement(SLIDER_KEY).Update(range=(1, len(all_frames))) 
        elif event == ANALYZE_KEY and video_loaded:
            print("analyze frames")
            (y_values, y_confidence) = analzye_frames(all_frames)
            y_values_strings = y_values_to_strings(label_names, y_values)

            annotated_frame_labels = get_y_values_for_gui(all_frames, y_values_strings)
             
            video_analyzed = True
            window.FindElement(LABEL_RESULTS_KEY).Update(values=annotated_frame_labels)
            # show plot
            fig = create_pyplot(y_values)  # if using Pyplot then get the figure from the plot
            delete_figure_agg(figure_canvas_agg)
            figure_canvas_agg = draw_figure(window[DISTRIBUTION_CANVAS_KEY].TKCanvas, fig)
        elif event == CLEAR_KEY:
            all_frames.clear()
            annotated_frame_labels.clear()
            y_confidence.clear()
            window[IMAGE_KEY].update(data=[])
            window.FindElement(LABEL_RESULTS_KEY).Update(values=[])
            video_loaded = False
            video_analyzed = False
        elif event == SLIDER_KEY and video_loaded:
            slider_val = int(values[SLIDER_KEY])
            # slider val starts at 1 so --> -1 to match array
            selected_frame = all_frames[slider_val-1]
            
            window[IMAGE_KEY].update(data=get_image_bytes(selected_frame))
        elif event == LABEL_RESULTS_KEY and video_analyzed:
            if len(values[LABEL_RESULTS_KEY]) > 0:
                 label_result_selection = values[LABEL_RESULTS_KEY][0]
                 frame_index = annotated_frame_labels.index(label_result_selection)
                 # we get index from an array so it already starts at 0 ;)   
                 selected_frame = all_frames[frame_index]
                 
                 # update frame
                 window[IMAGE_KEY].update(data=get_image_bytes(selected_frame))
                 
                 # update slider 
                 window.FindElement(SLIDER_KEY).Update(frame_index+1)
                 
                 # update confidence
                 window.FindElement(CONFIDENCE_OUTPUT_KEY).Update("")
                 text_format = ("Still: {:1.5f} \n" 
                 "Wave: {:1.5f} \n" 
                 "NearDrops: {:1.5f} \n" 
                 "SmallDrops: {:1.5f} \n" 
                 "Drops: {:1.5f} \n" 
                 "Foam: {:1.5f}")
                 
                 conf_values = y_confidence[frame_index]
                 text = text_format.format(conf_values[0], conf_values[1], 
                                           conf_values[2], conf_values[3], 
                                           conf_values[4], conf_values[5])
                 
                 window.FindElement(CONFIDENCE_OUTPUT_KEY).Update(text)
                 print(y_confidence[frame_index])
        elif event == SAVE_SESSION_PATH_KEY:
             selected_path = values[SAVE_SESSION_PATH_KEY]
             if selected_path != '':
                 df = pd.DataFrame(y_values, columns=['y_value'])
                 df_conf = pd.DataFrame(y_confidence, columns=LABELS)
                 concat = pd.concat([df, df_conf], axis=1)
                 concat.to_csv(selected_path, index=False)
        elif event == LOAD_SESSION_KEY and video_loaded:
             load_path = values[SESSION_BROWSE_KEY]
             session_df = pd.read_csv(load_path)
             
             if len(session_df) == len(all_frames):
                 y_values = session_df['y_value'].tolist()
                 y_confidence = session_df[LABELS].to_numpy().tolist()
                 
                 y_values_strings = y_values_to_strings(label_names, y_values)
                 annotated_frame_labels = get_y_values_for_gui(all_frames, y_values_strings)

                 video_analyzed = True
                 window.FindElement(LABEL_RESULTS_KEY).Update(values=annotated_frame_labels)
                 
                 fig = create_pyplot(y_values)  # if using Pyplot then get the figure from the plot
                 delete_figure_agg(figure_canvas_agg)
                 figure_canvas_agg = draw_figure(window[DISTRIBUTION_CANVAS_KEY].TKCanvas, fig)
                 
    window.close()

def create_pyplot(y_values):
    plt.clf() #clear plot
    # values_to_plot = [y_values.count(0), y_values.count(1), y_values.count(2), 
    #                   y_values.count(3), y_values.count(4), y_values.count(5)]
    values_to_plot = [y_values.count(i) for i in range(len(LABELS))]
    ind = np.arange(len(values_to_plot))
    width = 0.4 # bar plot width
   
    p1 = plt.bar(ind, values_to_plot, width)
    
    plt.xticks(ind, LABELS)
    plt.title("Frequency per Label")
    plt.ylabel("Frequency")
    plt.xlabel("Labels")
    
    # display values on bar
    for index, data in enumerate(values_to_plot):
        plt.text(x=index , y=data, s=f"{data}" , fontdict=dict(fontsize=10), ha='center')
    
    fig = plt.gcf()
    return fig
    

def get_y_values_for_gui(all_frames, y_values_strings):
    annotated_frame_labels = []
    for i in range(len(all_frames)): 
        text_format = "Frame {0}: {1}"
        text = text_format.format(i+1, y_values_strings[i])
        annotated_frame_labels.append(text)
        
    return annotated_frame_labels
                     

def get_image_bytes(frame):
    return cv2.imencode(".png", frame)[1].tobytes()

def y_values_to_strings(label_names, y_values):
    y_values_strings = []
    
    for y_value in y_values:
        y_values_strings.append(label_names[y_value])
    
    return y_values_strings

def analzye_frames(all_frames):
    y_values = []
    y_confidence = []
    normalized_frames = normalize_frames(all_frames)
    counter = 0
    for frame in normalized_frames:
        frame = frame.reshape(-1, frame.shape[0], frame.shape[1], 1)
        (y_val, prediction) = predict_instability(frame)
        y_values.append(y_val)
        y_confidence.append(prediction[0])        
        print("analyzing frame", counter)
        counter += 1
    
    return (y_values, y_confidence)

def draw_figure(canvas, figure, loc=(0, 0)):
    figure_canvas_agg = FigureCanvasTkAgg(figure, canvas)
    figure_canvas_agg.draw()
    figure_canvas_agg.get_tk_widget().pack(side='top', fill='both', expand=1)
    return figure_canvas_agg

# def create_figure(frame):
#     fig = matplotlib.figure.Figure(figsize=(5, 4), dpi=100)
#     fig.add_subplot(111)
#     fig.figimage(frame, resize=True)
#     return fig
            
def delete_figure_agg(figure_agg):
    if figure_agg:
        figure_agg.get_tk_widget().forget()
        plt.close('all')
    
def read_frames(path):
    vidcap = cv2.VideoCapture(path)
    vid_len = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    height = vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    width = vidcap.get(cv2.CAP_PROP_FRAME_WIDTH)
    
    frame_count = 0
    video_frames = []
    # read frames
    while True:
        success, frame = vidcap.read()
        if success:
            frame_count += 1
            print(str(frame_count)+'/' + str(vid_len) + ' h:' +str(int(height)) + ' w:'+str(int(width)), success)
            img_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            video_frames.append(img_gray)
        else:
            print("finished reading")
            break
    
    return video_frames

# def draw_figure(canvas, figure):
#     figure_canvas_agg = FigureCanvasTkAgg(figure, canvas)
#     figure_canvas_agg.draw()
#     figure_canvas_agg.get_tk_widget().pack(side="top", fill="both", expand=1)
#     return figure_canvas_agg

if __name__ == "__main__":
    main()