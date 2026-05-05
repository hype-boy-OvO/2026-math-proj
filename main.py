from models import ModifiedBert

model = ModifiedBert()

print("1. Pure Semantic (의미는 같으나 단어가 완전히 다른 경우)")
print(model.encode("The weather is lovely today."))
print(model.encode("It is a beautiful sunny day."))
print()

print("2. Functional Intent (의도와 목적이 동일한 경우)")
print(model.encode("I need a cup of coffee."))
print(model.encode("A shot of espresso would be great."))
print()

print("3. Lexical Trap (단어는 90% 일치하나 의미는 반대인 경우 - High 활성 인덱스 차이 확인용)")
print(model.encode("The movie was fantastic."))
print(model.encode("The movie was terrible."))
print()

print("4. Structural Change (능동태 vs 수동태 - 구조적 유사도 확인)")
print(model.encode("The cat chased the dog."))
print(model.encode("The dog was chased by the cat."))
print()

print("5. Polysemy (동음이의어 - 문맥에 따른 [CLS]의 변별력 확인)")
print(model.encode("I need to go to the bank for money."))
print(model.encode("We sat on the river bank to fish."))
print()

print("6. Verb Synonyms (행동의 유의어 처리 확인)")
print(model.encode("He ran to the store."))
print(model.encode("He sprinted toward the shop."))
print()

print("7. Tense & Grammar (시제 변화에 따른 활성화 민감도)")
print(model.encode("She is eating an apple."))
print(model.encode("She ate an apple."))
print()

print("8. Negation (부정어 하나가 전체 벡터를 얼마나 변화시키는지 확인)")
print(model.encode("I am feeling very happy."))
print(model.encode("I am not feeling very happy."))
print()

print("9. Technical & Abbreviation (약어와 전문 용어의 매칭 능력)")
print(model.encode("Artificial Intelligence is rising."))
print(model.encode("AI is growing fast."))
print()

print("10. Baseline - Unrelated (완전히 무관한 문장 - 유사도 판단의 최저 기준점 설정용)")
print(model.encode("The sky is blue."))
print(model.encode("I forgot my keys at home."))
print()
