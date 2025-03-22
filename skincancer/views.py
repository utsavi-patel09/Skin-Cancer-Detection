from django.shortcuts import render
from django.views import View
from django.core.files.storage import FileSystemStorage
import cv2
import numpy as np
from keras.models import load_model
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, 'models', 'A.h5')
model = load_model(model_path)

class Index(View):
    def get(self, request):
        return render(request, 'index.html', {"image1": "static/images/benign.jpg", "image2": "static/images/malignant.jpg"})

    def post(self, request):
        if request.method == 'POST' and request.FILES.get('file'):
            imagefile = request.FILES['file']
            fs = FileSystemStorage()
            image_path = fs.save(f'static/images/{imagefile.name}', imagefile)
            image_url = fs.url(image_path)
            image_full_path = os.path.join(fs.location, image_path)  # Ensure correct absolute path

            # Load the image
            img = cv2.imread(image_full_path)
            if img is None:
                return render(request, 'index.html', {
                    "error": "Invalid image file. Please upload a valid image format."
                })

            # Preprocess the image
            IMG_SIZE = 227
            new_array = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            new_array = new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 3)

            # Predict using the model
            x = model.predict(new_array)
            result = np.argmax(x)

            # Map result to labels
            if result == 0:
                predicted_result = "Benign"
                description="A benign skin tumor refers to a non-cancerous growth of skin cells. These tumors do not spread to other parts of the body and are generally not life-threatening. Common types of benign skin tumors include moles (nevi), lipomas, and seborrheic keratosis. Monitor Changes: Regularly check for changes in size, color, or texture of existing moles. Protection from Sun: Use sunscreen (SPF 30 or higher) to protect from harmful UV rays.Consult a Dermatologist: If you notice any growth or changes in your skin, even if it's benign, it's a good idea to get it checked. "
    
            else:
                predicted_result = "Malignant"
                description="Malignant skin cancer refers to cancerous growths that can spread (metastasize) to other parts of the body. The most common types of malignant skin cancers are melanoma, basal cell carcinoma (BCC), and squamous cell carcinoma (SCC).Precautions and Suggestions:Early Detection: Early detection of skin cancer greatly improves the chances of successful treatment. Perform regular self-exams and look for signs of asymmetry, irregular borders, uneven colors, or growth in existing moles.Avoid Excessive Sun Exposure: Stay out of the sun during peak hours (10 AM to 4 PM), wear protective clothing, and always apply sunscreen with broad-spectrum protection. Get Screened Regularly: If you have a family history of skin cancer or are at higher risk (e.g., fair skin, history of sunburns), schedule regular screenings with a dermatologist.Seek Medical Attention: If you notice any unusual or rapidly changing skin lesions, consult a healthcare provider immediately."
    

            return render(request, 'index.html', {
                'user_image': image_url,
                'predicted_result': predicted_result,
                'description': description
            })
        
        return render(request, 'index.html', {"error": "Please upload a valid image."})
