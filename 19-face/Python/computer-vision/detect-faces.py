from dotenv import load_dotenv
import os
from PIL import Image, ImageDraw
from matplotlib import pyplot as plt

# Import namespaces

 # import namespaces
import azure.ai.vision as sdk

def main():

    global cv_client

    try:
        # Get Configuration Settings
        load_dotenv()
        cog_endpoint = os.getenv('COG_SERVICE_ENDPOINT')
        cog_key = os.getenv('COG_SERVICE_KEY')

        # Authenticate Azure AI Vision client
        # Authenticate Azure AI Vision client
        cv_client = sdk.VisionServiceOptions(cog_endpoint, cog_key)


        # Detect faces in an image
        image_file = os.path.join('images','people.jpg')
        AnalyzeFaces(image_file)

    except Exception as ex:
        print(ex)

def AnalyzeFaces(image_file):
    print('Analyzing', image_file)

    # Specify features to be retrieved (faces)
    # Specify features to be retrieved (PEOPLE)
    analysis_options = sdk.ImageAnalysisOptions()
        
    features = analysis_options.features = (
        sdk.ImageAnalysisFeature.PEOPLE
    )    
    

    # Get image analysis
    # Get image analysis
    image = sdk.VisionSource(image_file)
        
    image_analyzer = sdk.ImageAnalyzer(cv_client, image, analysis_options)
        
    result = image_analyzer.analyze()
        
    if result.reason == sdk.ImageAnalysisResultReason.ANALYZED:
        # Get people in the image
        if result.people is not None:
            print("\nPeople in image:")
            
            # Prepare image for drawing
            image = Image.open(image_file)
            fig = plt.figure(figsize=(image.width/100, image.height/100))
            plt.axis('off')
            draw = ImageDraw.Draw(image)
            color = 'cyan'
            
            for detected_people in result.people:
                # Draw object bounding box if confidence > 50%
                if detected_people.confidence > 0.5:
                    # Draw object bounding box
                    r = detected_people.bounding_box
                    bounding_box = ((r.x, r.y), (r.x + r.w, r.y + r.h))
                    draw.rectangle(bounding_box, outline=color, width=3)
                
                    # Return the confidence of the person detected
                    print(" {} (confidence: {:.2f}%)".format(detected_people.bounding_box, detected_people.confidence * 100))
                        
            # Save annotated image
            plt.imshow(image)
            plt.tight_layout(pad=0)
            outputfile = 'detected_people.jpg'
            fig.savefig(outputfile)
            print('  Results saved in', outputfile)
        
    else:
        error_details = sdk.ImageAnalysisErrorDetails.from_result(result)
        print(" Analysis failed.")
        print("   Error reason: {}".format(error_details.reason))
        print("   Error code: {}".format(error_details.error_code))
        print("   Error message: {}".format(error_details.message))



if __name__ == "__main__":
    main()