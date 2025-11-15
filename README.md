1. Use just string matching for the task, but it is too slow.
2. Use .net for the api, but it is too heavy
3. Finally I decided to use python for the api and use LLM for question answering task.
4. For using LLM, use RAG is a good idea to save the time for answering the questions for the same piece of data

For the data:
All 3,349 entries have complete records for both user_id and user_name. There are no missing or null values in the key member identification fields.
There is no inconsistency found where a single user_id is associated with multiple different user_names. 
Every unique user_id maps to a single, unique user_name, indicating a clean and reliable mapping between the user identifier and the displayed name.