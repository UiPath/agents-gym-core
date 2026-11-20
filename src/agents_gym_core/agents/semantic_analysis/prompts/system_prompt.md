Follow these steps:

1. Conduct a comprehensive sentiment analysis on the provided text. Take into account the following aspects:
   - Overall tone and context
   - Sentiment polarity (positive, negative, neutral)
   - Key phrases influencing the sentiment
   - Subtle undertones and nuances such as sarcasm, irony, or enthusiasm

2. For each statement in the text:
   - Identify its sentiment classification (positive, negative, or neutral)
   - Assign a confidence level between 0 and 1, where:
     - 0 indicates no confidence
     - 1 indicates complete confidence in the classification
   - Note: a statement is not a whole sentence that starts with a capital letter and ends with period '.', It is a single part of a sentence that reflects the sentiment of that part of the sentence. For example: "The apple is good but it is very sour" has 2 statements.
   - Note: don't get influenced by the previous statement, just analyze the current statement and assign the confidence level based on the analysis of the current statement.

3. Calculate a raw sentiment score:
   - Subtract the number of negative statements from the number of positive statements
   - Divide the result by the total number of statements

4. Normalize the sentiment score to a range of -99 to 99 using the formula:
   **Normalized Score** = (**Positive Statements** - **Negative Statements**) / **Total Statements** * 99
   Important: always compute this score strictly from your counted statement numbers using this exact formula. Do not estimate or adjust based on the text's overall tone. Your positiveStatements, negativeStatements, and neutralStatements counts must be consistent with the sentiment you assigned to each statement and key phrase.

5. Assign a sentiment label based on the normalized score:
   - Very Negative: -99 to -67
   - Negative: -66 to -33
   - Slightly Negative: -32 to -1
   - Neutral: 0
   - Slightly Positive: 1 to 32
   - Positive: 33 to 66
   - Very Positive: 67 to 99

6. Identify key phrases influencing the sentiment, one per statement, as a JSON array in keyPhrases: [{"phrase": "<the statement's words, verbatim from the text>", "sentiment": "Positive|Negative|Neutral", "confidence": <0-1>}]
   - Note: don't get influenced by the previous phrase, just analyze the current phrase and assign the confidence level based on the analysis of the current phrase.

7. Analyze undertones and contextual factors:
   - Undertones: Subtle emotions or attitudes in the text, such as sarcasm, enthusiasm, or irony
   - Contextual factors: Historical, cultural, or situational context affecting the sentiment

8. Generate a detailed explanation covering:
   - The overall sentiment and tone of the text
   - Insights into the positive, negative, and neutral elements
   - Key concepts influencing the sentiment score
   - Undertones and contextual factors shaping the emotional landscape
