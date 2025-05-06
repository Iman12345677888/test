# from transformers import pipeline
# import numpy as np
#
#
# def answer_land_cover_question(question: str, indices: dict) -> str:
#     """
#     Answer a land cover question using spectral indices and an LLM.
#
#     Args:
#         question (str): User's question about land cover.
#         indices (dict): Dictionary of spectral indices (e.g., {'NDVI': array, 'NDWI': array}).
#
#     Returns:
#         str: Natural language answer.
#     """
#     # Summarize indices (e.g., mean values)
#     ndvi_mean = np.mean(indices['NDVI'])
#     ndwi_mean = np.mean(indices['NDWI'])
#
#     # Create prompt
#     prompt = f"""
#     Question: {question}
#     Context: The region has an average NDVI of {ndvi_mean:.2f} and NDWI of {ndwi_mean:.2f}.
#     NDVI indicates vegetation health (higher values mean healthier vegetation).
#     NDWI indicates water content (higher values mean more water presence).
#     Provide a concise answer based on this data.
#     """
#
#     # Initialize LLM (replace with your preferred model/API)
#     nlp = pipeline("text-generation", model="distilbert-base-uncased")
#     response = nlp(prompt, max_length=100, num_return_sequences=1)
#
#     return response[0]['generated_text'].strip()


import litellm
import numpy as np


def answer_land_cover_question(question: str, indices: dict) -> str:
    """
    Answer a land cover question using spectral indices and an LLM.

    Args:
        question (str): User's question about land cover.
        indices (dict): Dictionary of spectral indices (e.g., {'NDVI': array, 'NDWI': array}).

    Returns:
        str: Natural language answer.
    """
    # ndvi_mean = np.mean(indices['NDVI'])
    # ndwi_mean = np.mean(indices['NDWI'])
    # {
    #     "NDVI": -0.021398334389402404,
    #     "NDWI": 0.011193134274381152,
    #     "EVI": null,
    #     "SAVI": -0.028579522081362434,
    #     "NDBI": -0.6896037361297463,
    #     "NDMI": 0.6896037361297463,
    #     "ARVI": -0.0019433430125375675,
    #     "GNDVI": -0.011193134274381152,
    #     "SIPI": null,
    #     "ExG": -0.004401347351074218,
    #     "ExR": 0.25205025577545165,
    #     "ExGR": -0.25645160312652593,
    #     "VARI": -0.016018050163049052
    # }

    prompt = f"""
    Question: {question}
    Context: The region has an average NDVI of {indices["NDVI"]} and NDWI of {indices["NDWI"]}
    and EVI of {indices["EVI"]} and SAVI of {indices["SAVI"]} and NDBI of {indices["NDBI"]}
    and NDMI of {indices["NDMI"]} and ARVI of {indices["ARVI"]} and GNDVI of {indices["GNDVI"]}
    and SIPI of {indices["SIPI"]} and ExG of {indices["ExG"]} and ExR of {indices["ExR"]}
    and ExGR of {indices["ExGR"]} and VARI of {indices["VARI"]}.
    
    Provide a concise answer based on this data.
    """

    response = litellm.completion(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a land cover analysis expert."},
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message.content.strip()