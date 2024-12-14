from openai import OpenAI
import csv

client = OpenAI(api_key = "")

sys_prom = "You are a therapeutic consultant, you do not make direct and strong statements to patients."

f = open("DirectStatements.csv", "r", encoding='utf-8-sig')


with open("A4_4_4.csv", mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Input statement", "Produced output"])

    for i in f.readlines():
        sentence=i.strip()
        usr_prom= "\""+sentence+"\"into more softened version, make the sentence less determined, less certain, and leave room for the reciever to make the conclusion by themselves."

        completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": sys_prom},
            {
                "role": "user",
                "content": usr_prom
            }
        ]
        )

        generated_output = completion.choices[0].message.content
        print(generated_output)
        writer.writerow([sentence, generated_output])

f.close()


