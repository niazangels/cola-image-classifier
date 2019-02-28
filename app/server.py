from starlette.applications import Starlette
from starlette.responses import HTMLResponse, JSONResponse
from starlette.staticfiles import StaticFiles
from starlette.middleware.cors import CORSMiddleware
import uvicorn, aiohttp, asyncio
from io import BytesIO

from fastai import *
from fastai.vision import *

#model_file_url = 'https://www.dropbox.com/s/xj9hzyytbip8ka7/stage-2-299-rn50.pth?raw=1'
model_file_name = 'model1'
#model_file_url = 'https://www.dropbox.com/s/y4kl2gv1akv7y4i/stage-2.pth?raw=1'
#model_file_name = 'model'
#classes = ['coca_cola', 'nuka_cola', 'other', 'pepsi_cola']
#nice_labels = ['Coca Cola', 'Nuka Cola', 'Something Else', 'Pepsi Cola']
size_big = 299
#path = Path(__file__).parent
path = Path("/home/jupyter/.fastai/data/notes/")
defaults.device = torch.device('cpu')
    
app = Starlette()
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_headers=['X-Requested-With', 'Content-Type'])
app.mount('/static', StaticFiles(directory='app/static'))
#app.wsgi_app = ReverseProxied(app.wsgi_app)

#async def download_file(url, dest):
#    if dest.exists(): return
#    async with aiohttp.ClientSession() as session:
#        async with session.get(url) as response:
#            data = await response.read()
#            with open(dest, 'wb') as f: f.write(data)

CLASSES = []
async def setup_learner():
    #await download_file(model_file_url, path/'models'/f'{model_file_name}.pth')
    global CLASSES
    #data2 = ImageDataBunch.single_from_classes(path, CLASSES, size=size_big).normalize(imagenet_stats)
    data = ImageDataBunch.from_folder(path, 
                                  train=path / 'train', 
                                  valid_pct=0.2,
                                  ds_tfms=get_transforms(), 
                                  size=224, 
                                  num_workers=4
                                 ).normalize(imagenet_stats)    
    CLASSES = data.classes
    learn = create_cnn(data, models.resnet34)
    learn.load(model_file_name)
    return learn

loop = asyncio.get_event_loop()
tasks = [asyncio.ensure_future(setup_learner())]
learn = loop.run_until_complete(asyncio.gather(*tasks))[0]
loop.close()

@app.route('/')
def index(request):
    html = Path(__file__).parent / 'view' / 'index.html'
    return HTMLResponse(html.open().read())

@app.route('/analyze', methods=['POST'])
async def analyze(request):
    data = await request.form()
    img_bytes = await (data['file'].read())
    img = open_image(BytesIO(img_bytes))
    pred_class,pred_idx,outputs = learn.predict(img)
    formatted_outputs = ["{:.1f}%".format(value) for value in [x * 100 for x in torch.nn.functional.softmax(outputs, dim=0)]]
    pred_probs = sorted(
            zip(CLASSES, map(str, formatted_outputs)),
            key=lambda p: p[1],
            reverse=True
        )
    return JSONResponse({'result': pred_probs[0]})

if __name__ == '__main__':
    if 'serve' in sys.argv: uvicorn.run(app, host='0.0.0.0', port=5041)
