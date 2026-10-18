from micronmt.api import Translator

translator = Translator('runs/tiny/best.pt', terminology={
    'deployment': ['развёртывание', 'деплой'],
})
print(translator.translate(
    'I decided to stay there.', source='en', target='ru',
    context='I booked a room in a small hotel.',
))
