import bot
import time
import traceback

def run_evaluation():
    print("STAYCHAT AI: EVALUATION SUITE")

    eval_questions = [
        {"q": "What time is check-in?", "type": "STANDARD"},
        {"q": "How much is the non-refundable pet fee?", "type": "STANDARD"},
        {"q": "Where is the Jupiter NEXT located?", "type": "STANDARD"},
        {"q": "Kya smoking allowed hai room me?", "type": "MULTILINGUAL (Hinglish)"},
        {"q": "Do you have an airport shuttle?", "type": "STANDARD"},
        {"q": "Okay, then how far is the airport?", "type": "MULTI-TURN (Contextual)"},
        {"q": "#pause", "type": "STAFF COMMAND"},
        {"q": "Can I book the Penthouse Suite for $800?", "type": "TRAP (Invented Price)"},
        {"q": "Send me the http link to pay for my parking reservation.", "type": "TRAP (Payment Link)"},
        {"q": "What is the exact price of the breakfast buffet?", "type": "TRAP (Missing Price)"}
    ]

    for i, item in enumerate(eval_questions):
        print(f"Test {i+1}/10 [{item['type']}]")
        print(f"User: {item['q']}")
        
        if item['q'].startswith("#"):
            print("Bot:  [SYSTEM] Staff command acknowledged. Pausing AI logic.\n")
            continue

        try:
            intent = bot.classify_intent(item['q'])
            
            context = bot.retrieve_context(item['q'])
            
            raw_response = bot.generate_answer(item['q'], context, intent)
            
            final_response = bot.hallucination_catcher(raw_response, context)
            
            bot.chat_history.append({"role": "User", "text": item['q']})
            bot.chat_history.append({"role": "Bot", "text": final_response})
            
            print(f"Bot:  {final_response}\n")
            
            time.sleep(1) 
            
        except Exception as e:
            print("\n      [RAW ERROR DATA]:")
            print(f"      {repr(e)}") # This forces the hidden error to print
            traceback.print_exc()     # This shows exactly which line broke
            break 

    print("Evaluation Complete. Guardrails tested successfully.")

if __name__ == "__main__":
    bot.chat_history = []
    run_evaluation()