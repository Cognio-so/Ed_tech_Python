import openai
import os
import requests
from PIL import Image
from io import BytesIO
from dotenv import load_dotenv
load_dotenv()

# --- OpenAI API Initialization ---
# It's recommended to set your API key as an environment variable for security.
# On Mac/Linux: export OPENAI_API_KEY='your-key'
# On Windows: set OPENAI_API_KEY='your-key'
try:
    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
except TypeError:
    print("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
    exit()

def get_user_input():
    """
    Gets the comic book topic, target audience, and number of panels from the user.
    """
    print("Welcome to the AI Comic Generator!")
    instructions = input("What educational topic would you like to make a comic about? (e.g., 'The water cycle')\n> ")
    student_class = input("What grade level is this for? (e.g., '3rd grade')\n> ")
    language = input("What language should the comic be in? (e.g., 'English', 'Arabic')\n> ")
    while True:
        try:
            num_panels = int(input("How many panels should the comic have? (e.g., 4)\n> "))
            if num_panels > 0:
                break
            else:
                print("Please enter a positive number for the panels.")
        except ValueError:
            print("Invalid input. Please enter a number.")
    return instructions, student_class, num_panels, language

def create_comical_story_prompt(instructions, student_class, num_panels, language):
    """
    Uses GPT-4o to turn the user's instructions into a series of detailed prompts for GPT Image 1,
    based on the specified number of panels and target audience and language.
    """
    print("\nTurning your idea into a fun comic story...")
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a creative storyteller for children's educational comics. "
                        "Your task is to take a user's topic and create a fun, educational comic story. "
                        "The story should be tailored for a specific grade level and have a specific number of panels, as requested by the user. "
                        f"Generate all dialogue and narrative text in {language}. "
                        "For each panel, you must provide a single, detailed 'Panel_Prompt' for an image generation model (gpt-image-1). "
                        "This prompt must describe the entire scene, the characters, their actions, and include the exact dialogue or narrative text that should appear in speech bubbles or captions within the image. "
                        "The language and complexity of the topic should be appropriate for the target students. "
                        f"Ensure the visual style is described as fun, kid-friendly, and colorful. based on {student_class} class students "
                        "Structure the output as a numbered list of prompts, one for each panel requested. Start each line with 'Panel_Prompt:'. "
                        "For example: 'Panel_Prompt: A friendly water drop character smiling in a cloud, with a speech bubble that says, \"I'm getting heavy! Time to fall as rain!\"'"
                        "Follow same animation throughout the storyline"
                    )
                },
                {
                    "role": "user",
                    "content": (
                        f"Create a comic story with the following details:\n"
                        f"- Topic: {instructions}\n"
                        f"- Target Audience: {student_class} class students\n"
                        f"- Number of Panels: {num_panels}"
                        f"- Language: {language}"
                    )
                }
            ],
            temperature=0.8,
        )
        story_prompts = response.choices[0].message.content
        return story_prompts
    except openai.APIError as e:
        print(f"An error occurred with the OpenAI API: {e}")
        return None

def generate_comic_image(prompt, panel_number):
    """
    Uses gpt-image-1 to generate a comic book panel image based on the prompt.
    The prompt should include instructions for text to be rendered in the image.
    """
    print(f"Generating image for panel {panel_number}...")
    try:
        response = client.images.generate(
            model="gpt-image-1",
            prompt=prompt,
            size="1024x1024",
            quality="high",
            n=1,
        )
        image_base64 = response.data[0].b64_json
        return image_base64
    except openai.APIError as e:
        print(f"An error occurred with the image generation API: {e}")
        return None

def main():
    """
    Main function to run the comic generator.
    """
    instructions, student_class, num_panels, language = get_user_input()
    if not instructions:
        print("No instructions provided. Exiting.")
        return

    comical_story_prompts = create_comical_story_prompt(instructions, student_class, num_panels,language)
    if not comical_story_prompts:
        print("Could not generate a story. Exiting.")
        return

    print("\n--- Your Comic Story Prompts ---\n")
    print(comical_story_prompts)
    print("\n--- Generating Comic Panel Images ---\n")

    # The story prompts are split by newline characters.
    panel_prompts = [line.split("Panel_Prompt:")[1].strip() for line in comical_story_prompts.strip().split('\n') if "Panel_Prompt:" in line]
    
    if not panel_prompts:
        print("Could not parse the story prompts. Please check the output from the story generation.")
        return
        
    # Warn the user if the number of generated panels does not match the requested number
    if len(panel_prompts) != num_panels:
        print(f"Warning: The AI generated {len(panel_prompts)} panels, but {num_panels} were requested. Proceeding with the generated panels.")

    for i, prompt in enumerate(panel_prompts):
        panel_number = i + 1
        print(f"\n--- Panel {panel_number} ---")
        
        image_base64 = generate_comic_image(prompt, panel_number)
        
        if image_base64:
            print(f"Comic Panel {panel_number} (b64_json):\n{image_base64}")
        else:
            print("Failed to generate image for this panel.")

if __name__ == "__main__":
    main()