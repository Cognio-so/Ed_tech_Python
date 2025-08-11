import os
import time
import requests
from dotenv import load_dotenv

load_dotenv()

class SlideSpeakGenerator:
    """
    A class to generate and retrieve SlideSpeak presentations.
    """
    def __init__(self, api_key=None):
        """
        Initializes the SlideSpeakGenerator.

        Args:
            api_key (str, optional): The SlideSpeak API key. If not provided,
                                     it's retrieved from the SLIDESPEAK_API_KEY
                                     environment variable.
        """
        self.api_key = os.getenv("SLIDESPEAK_API_KEY")
        if not self.api_key:
            raise ValueError("SLIDESPEAK_API_KEY not provided or set as an environment variable.")
        self.base_url = "https://api.slidespeak.co/api/v1"

    def generate_presentation(
        self,
        plain_text: str,
        custom_user_instructions: str,
        length: int,
        language: str = "ENGLISH",
        fetch_images: bool = True,
        verbosity: str = "standard",
        tone: str = "educational"
    ):
        """
        Generates and retrieves a SlideSpeak presentation by polling the task status.

        Args:
            plain_text (str): The main topic or plain_text of the presentation.
            custom_user_instructions (str): Specific instructions for the AI.
            length (int): The desired number of slides.
            language (str, optional): The language of the presentation. Defaults to "ENGLISH".
            fetch_images (bool, optional): Whether to include stock images. Defaults to True.
            verbosity (str, optional): The desired text verbosity. Defaults to "standard".
            tone (str, optional): The tone of the presentation. Defaults to "educational".

        Returns:
            dict: The final JSON response from the SlideSpeak API.
        """
        headers = {
            "Content-Type": "application/json",
            "X-API-Key": self.api_key,
        }
        payload = {
            "plain_text": plain_text,
            "custom_user_instructions": custom_user_instructions,
            "length": length,
            "language": language,
            "fetch_images": fetch_images,
            "verbosity": verbosity,
            "tone": tone
        }

        try:
            initial_response = requests.post(f"{self.base_url}/presentation/generate", headers=headers, json=payload)
            initial_response.raise_for_status()
            initial_data = initial_response.json()

            if "task_id" not in initial_data:
                return {"error": "task_id not found in initial response", "details": initial_data}

            task_id = initial_data["task_id"]
            print(f"Presentation generation started with task_id: {task_id}")

            status_url = f"{self.base_url}/task_status/{task_id}"
            while True:
                print("Checking task status...")
                status_response = requests.get(status_url, headers=headers)
                status_response.raise_for_status()
                status_data = status_response.json()

                task_status = status_data.get("task_status")

                if task_status == "SUCCESS":
                    print("Presentation generated successfully!")
                    return status_data
                elif task_status == "FAILURE":
                    print("Presentation generation failed.")
                    return status_data
                else:
                    print(f"Status is '{task_status}'. Waiting for 5 seconds...")
                    time.sleep(5)

        except requests.exceptions.RequestException as e:
            return {"error": str(e)}

def main():
    """
    Main function to run the SlideSpeak presentation generator from the command line.
    """
    print("--- SlideSpeak Presentation Generator ---")
    print("Please provide the following details for your presentation:")

    presentation_plain_text = input("Plain Text: ")
    user_instructions = input("Custom Instructions: ")
    num_slides = int(input("Number of Slides: "))
    presentation_language = input("Language (e.g., ENGLISH, ARABIC): ").upper()
    fetch_images_input = input("Fetch Images (true/false): ").lower()
    presentation_verbosity = input("Verbosity (concise/standard/text-heavy): ").lower()
    presentation_tone = "educational"

    fetch_images_bool = fetch_images_input == "true"

    generator = SlideSpeakGenerator()

    print("\nGenerating your presentation...")
    final_result = generator.generate_presentation(
        plain_text=presentation_plain_text,
        custom_user_instructions=user_instructions,
        length=num_slides,
        language=presentation_language,
        fetch_images=fetch_images_bool,
        verbosity=presentation_verbosity,
        tone=presentation_tone
    )

    print("\n--- Final API Response ---")
    print(final_result)

if __name__ == "__main__":
    main()