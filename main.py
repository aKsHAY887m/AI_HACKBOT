# main.py

from ai_core import AdaptiveEngine


MODEL_FILE = "brain_state.joblib"



def show_status(ai):

    print("\n======================")

    visible = ai.visible_history()

    if visible:
        print(
            "Last 10 outcomes:",
            " ".join(visible)
        )

    else:
        print(
            "Last 10 outcomes: Empty"
        )


    print(
        "Total observations:",
        len(ai.history)
    )

    print("======================")



def show_prediction(ai):

    result = ai.predict()


    prediction = result["prediction"]

    confidence = result["confidence"]

    explanation = result["explanation"]



    if prediction is None:

        print(
            "Prediction: Not enough data"
        )

        return



    print(
        "\nPrediction:",
        prediction
    )


    print(
        "Confidence:",
        f"{confidence*100:.2f}%"
    )


    print(
        "Explanation:"
    )


    for item in explanation:

        print(
            "-",
            item
        )



def main():


    ai = AdaptiveEngine.load(
        MODEL_FILE
    )


    print(
        "=== Adaptive Multi-Brain Sequence AI ==="
    )


    if ai.history:

        print(
            "Previous memory loaded."
        )

    else:

        print(
            "Starting with empty memory."
        )


    while True:


        show_status(ai)


        user = input(
            "\nEnter A/B (or predict/save/exit): "
        ).strip().upper()



        if user == "EXIT":

            ai.save(
                MODEL_FILE
            )

            print(
                "Saved. Exiting."
            )

            break



        elif user == "SAVE":

            ai.save(
                MODEL_FILE
            )

            print(
                "Memory saved."
            )

            continue



        elif user == "PREDICT":

            show_prediction(ai)

            continue



        elif user in ["A","B"]:


            # Predict before seeing new data

            show_prediction(ai)


            # Now learn the actual observation

            ai.learn(
                user
            )


            print(
                "Learned:",
                user
            )


            if ai.drift.detected():

                print(
                    "Warning: Possible pattern drift detected."
                )



        else:

            print(
                "Invalid input. Use A, B, predict, save, or exit."
            )



if __name__ == "__main__":

    main()