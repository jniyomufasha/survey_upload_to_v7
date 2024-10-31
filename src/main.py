import os
import boto3
from io import BytesIO
from ultralytics import YOLO
import tempfile
import shutil
from darwin.client import Client
import json
import concurrent.futures
import pandas as pd

from darwin.importer import importer, get_importer

THIS_FILE_DIR = os.path.dirname(os.path.abspath(__file__))
WORK_DIR = os.path.normpath(os.path.join(THIS_FILE_DIR, '../data'))
MODEL_RELATIVE_PATH = 'models/best.pt' #Relative to WORK_DIR

BUCKET_NAME = 'zipline-photogrammetry-service-kms'
V7_DATASET_SLUG = 'tower-batch-7'
IMAGE_PREFIX = 'photogrammetry_service/images/'
MODEL_CONF_THRESHOLD = 0.4

model = YOLO(os.path.join(THIS_FILE_DIR,'models/best.pt'))
s3 = boto3.client('s3')
parser = get_importer("darwin")

os.makedirs(WORK_DIR, exist_ok= True)
os.makedirs(os.path.join(WORK_DIR, 'images_with_towers'), exist_ok= True)
os.makedirs(os.path.join(WORK_DIR, 'output_files'), exist_ok= True)
os.makedirs(os.path.join(WORK_DIR, 'output_files', 'annotations'), exist_ok= True)

with open(os.path.normpath(os.path.join(THIS_FILE_DIR, '../secrets.json'))) as f:
    V7_API_KEY = json.load(f)['v7_api_key']
client = Client.from_api_key(V7_API_KEY)
dataset = client.get_remote_dataset(dataset_identifier=f'zipline/{V7_DATASET_SLUG}')


projects_to_process = [
    'Project_ci1_flight_35808',
    'Project_ci1_flight_40861',
    ]

def extract_image_keys_to_process(project):
    image_json_file_key = IMAGE_PREFIX + f"{project}/images.json"
    obj = s3.get_object(Bucket = BUCKET_NAME, Key = image_json_file_key)
    images = json.loads(obj["Body"].read())
    return images

def save_annotations_locally(image_name, image_path, model_results):
    assert len(model_results[0].boxes) >= 1, "The passed model results should contain at least one box"
    annotation_dict = {
        "version": "2.0",
        "item": {
            "name": image_name,
            "path": image_path,
        },
        "annotations": []
    }

    for box in model_results[0].boxes:
        x,y,w,h = box.xywh.tolist()[0]
        annotation_dict['annotations'].append({
            "bounding_box": {
                "h": h,
                "w": w,
                # While experimenting, I found that the x and y outputted by he model are the center of the box
                # yet darwin expects x and y as the top left corner. So. I will make the adjustments below
                "x": x - (w/2),
                "y": y - (h/2)
            },
            "name": "Tower"
        })
    
    json_str = json.dumps(annotation_dict, indent= 4)
    json_file_path = os.path.join(WORK_DIR, 'output_files', 'annotations', f"{image_name.rsplit('.', 1)[0]}.json")
    with open(json_file_path, 'w') as f:
        f.write(json_str)

def process_results(results):
    result = results[0]
    if len(result.boxes) > 0:
        return 1
    return 0


def run_model_on_stream(stream, image_name = '', project = ''):
    detect_flag = False  # Initialize detect_flag
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:  # Ensure manual control over deletion
        stream.seek(0)
        shutil.copyfileobj(stream, temp_file)  # Copy stream content to temp file
        temp_file.flush()  # Ensure data is written before reading

        if not os.path.exists(temp_file.name):
            print(f"File {temp_file.name} not found before model inference. {image_name}")

        renamed_image_path = None

        try:
        # Model inference
            results = model(temp_file.name, conf=MODEL_CONF_THRESHOLD, verbose = False)  # Adjust according to your model's API
            detect_flag = process_results(results)
            if detect_flag: # Tower found. Save the image and annotations locally
                new_image_path = shutil.copy(temp_file.name, os.path.join(WORK_DIR, 'images_with_towers'))
                renamed_image_path = os.path.join(WORK_DIR, 'images_with_towers', image_name)
                os.rename(new_image_path, renamed_image_path)
                save_annotations_locally(image_name = image_name, image_path = f"/{project}/", model_results = results)
        except Exception as e:
            print(f"The image {image_name} was not processed. Error: {e}")
        finally:
            # Clean up the temporary file
            os.remove(temp_file.name)
        print(f'Done with {image_name}. Tower detected: {detect_flag}')
    return detect_flag, results, renamed_image_path

def process_image(key):
    file_stream = BytesIO()
    s3.download_fileobj(BUCKET_NAME, key, file_stream)
    image_name = key.replace(IMAGE_PREFIX, '').replace('/', '.')
    project = key.replace(IMAGE_PREFIX, '').split('/')[0]
    detect_flag, results, new_image_path = run_model_on_stream(file_stream, image_name, project = project)

    return {
        'key': key,
        'project': project,
        'tower_detected': detect_flag,
        'image_path': new_image_path,
    }

def upload_project(results_df_with_towers, project, dataset = dataset):
    assert list(results_df_with_towers['tower_detected'].unique()) == [True]
    assert project in results_df_with_towers['project'].unique()
    this_project_df = results_df_with_towers.query('project == @project')
    paths = list(this_project_df['image_path'].unique())
    print(f'Going to push {len(this_project_df)} images from {project}')
    try:
        dataset.push(files_to_upload = paths, path = f"/{project}/")
        print(f'Successfully pushed {project}')
    except Exception as e:
        print(f'pushing {project} failed. Error: {e}')

def upload_annotations(results_df_with_towers, project, dataset = dataset):
    assert list(results_df_with_towers['tower_detected'].unique()) == [True]
    assert project in results_df_with_towers['project'].unique()
    this_project_df = results_df_with_towers.query('project == @project')
    annotation_paths = []
    for image_path in this_project_df['image_path'].unique():
        image_name = image_path.split('/')[-1]
        annotation_file_name = image_name.rsplit('.', 1)[0]+'.json'
        annotation_paths.append(os.path.join(WORK_DIR, 'output_files', 'annotations', annotation_file_name))
    print(f'Going to push annotations images to {project}')
    importer.import_annotations(dataset, parser, annotation_paths, append=True)
    print(f'Successfully pushed annotations to {project}')
    


with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
    futures = [executor.submit(extract_image_keys_to_process, project) for project in projects_to_process]
    results = [future.result() for future in concurrent.futures.as_completed(futures) if future.result() != {}]

all_given_keys = [key for project_keys in results for key in project_keys]
keys_to_process = all_given_keys # In the future, we would implement skipping the ones that had been run before

print(f'About to process {len(keys_to_process)} images')
with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
    futures = [executor.submit(process_image, key) for key in keys_to_process]
    results = [future.result() for future in concurrent.futures.as_completed(futures) if future.result() != {}]

results_df = pd.DataFrame(results)
for project in results_df['project'].unique():
        results_df.query('project == @project').to_csv(os.path.join(WORK_DIR, 'output_files', f'{project}_results.csv'))
tower_df = results_df.query('tower_detected == True')
projects_without_towers = list(set(results_df['project'].unique()) - set(tower_df['project'].unique()))
if len(projects_without_towers) > 0:
    print('\n\nNo towers found in\n', projects_without_towers)
if len(tower_df) > 0:
    for project in tower_df['project'].unique():
        print(f"project has {len(tower_df.query('project == @project'))} images with towers")
        upload_project(tower_df, project)
        # upload_annotations(tower_df, project) #Uploading annotations right after uploading images is not working because the images are still processing.