"""Single-host educational chest X-ray classifier. Run `python app.py --help`."""
import argparse
import base64
from email.parser import BytesParser
from email.policy import default
import io
import json
import sys
import threading
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from PIL import Image, ImageOps, UnidentifiedImageError

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "model.pt"
MAX_UPLOAD = 10 * 1024 * 1024
Image.MAX_IMAGE_PIXELS = 20_000_000

PAGE = '''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Chest X-ray study prototype</title><style>
:root{font-family:system-ui,-apple-system,Segoe UI,sans-serif;color:#18334a;background:#eff5f8}*{box-sizing:border-box}
body{margin:0}header{background:#102f49;color:white;padding:26px max(5vw,18px)}header h1{margin:0;font-size:clamp(1.5rem,3vw,2.5rem)}header p{margin:8px 0 0;color:#c8dfea}
main{max-width:1050px;margin:30px auto;padding:0 18px}.card{background:white;border-radius:18px;padding:26px;box-shadow:0 10px 30px #163a5211;margin-bottom:20px}
.badge{display:inline-block;background:#fff1cc;color:#775000;border-radius:99px;padding:7px 12px;font-weight:700}.warning{border-left:4px solid #ca7030;padding:11px 15px;background:#fff7ec;border-radius:8px;margin:20px 0}
.row{display:grid;grid-template-columns:1fr 1fr;gap:22px}@media(max-width:700px){.row{grid-template-columns:1fr}}
input[type=file]{width:100%;border:2px dashed #9db8c8;border-radius:12px;padding:30px 16px;background:#f7fbfd}button{background:#007e91;color:white;border:0;border-radius:10px;padding:12px 22px;font-size:1rem;font-weight:700;cursor:pointer;margin-top:16px}button:disabled{opacity:.55;cursor:wait}
img{max-width:100%;max-height:390px;object-fit:contain;border-radius:10px;background:#e6edf0}#preview,#heatmap{display:none}#result{min-height:80px;line-height:1.6}small{color:#476276}
.finding-list{list-style:none;padding:0;margin:12px 0}.finding-list li{display:flex;justify-content:space-between;gap:16px;border-bottom:1px solid #e3ebef;padding:9px 0}.finding-list strong{white-space:nowrap}
</style></head><body><header><h1>Chest X-ray study prototype</h1><p>One Python server · Upload · Classification · Grad-CAM</p></header><main>
<section class="card"><span class="badge" id="status">Checking model…</span><div class="warning"><strong>Educational research prototype.</strong> A model score is not a medical diagnosis or a calibrated probability. A low pneumonia score cannot rule out pneumonia or any other disease. Seek a qualified clinician's assessment for any health concern. On a public demo, upload only de-identified example images.</div>
<h2>Analyze an image</h2><p>Choose a chest X-ray in JPG or PNG format (up to 10 MB). Uploaded images are processed in memory and are not saved by this app.</p>
<input type="file" id="file" accept="image/png,image/jpeg"><br><button id="analyze">Analyze X-ray</button><p id="message" role="alert"></p></section>
<section class="card row"><div><h2>Uploaded image</h2><img id="preview" alt="Uploaded chest X-ray"><p id="empty">Your image appears here.</p></div><div><h2>Model output</h2><div id="result">Choose a chest X-ray and click Analyze X-ray.</div><div id="findings"></div><img id="heatmap" alt="Grad-CAM model attention overlay"><p><small id="heatnote"></small></p></div></section>
<section class="card"><h2>How to start</h2><p>Run <code>python app.py serve</code>. Published pretrained weights are included. To train your own binary model, place labelled images under <code>dataset/train/NORMAL</code>, <code>dataset/train/PNEUMONIA</code>, <code>dataset/val/NORMAL</code>, and <code>dataset/val/PNEUMONIA</code>. Run <code>python app.py train</code>, then restart <code>python app.py serve</code>.</p></section>
<section class="card" style="background:#102f49;color:white;text-align:center">
<h2 style="margin-top:0">PNEUMONIA DIAGNOSIS SYSTEM USING DEEP LEARNING TECHNIQUES</h2>
<p style="color:#c8dfea">Developed by</p>
<p><strong>Meruva Harthik</strong><br>(24UECS1291) · (VTU27467)</p>
<p><strong>Nujendla Venu</strong><br>(24UECS1318) · (VTU28482)</p>
<p><strong>Peroori Gowtham Surya Manikanta</strong><br>(24UECS1332) · (VTU28294)</p>
<p style="color:#c8dfea;margin-top:24px">Under the guidance of</p>
<p style="margin-bottom:0"><strong>Dr. Sendil Vadivu D., M.E., Ph.D.</strong><br>Assistant Professor</p>
</section>
</main><script>
const f=document.getElementById('file'), b=document.getElementById('analyze'), msg=document.getElementById('message');
fetch('/api/health').then(r=>r.json()).then(x=>{document.getElementById('status').textContent=x.model_loaded?'Model ready: '+x.mode:'Model not loaded'});
f.onchange=()=>{const p=document.getElementById('preview');if(f.files[0]){p.src=URL.createObjectURL(f.files[0]);p.style.display='block';document.getElementById('empty').style.display='none';}};
b.onclick=async()=>{if(!f.files[0]){msg.textContent='Choose a JPG or PNG image first.';return}b.disabled=true;msg.textContent='Analyzing…';document.getElementById('heatmap').style.display='none';document.getElementById('findings').replaceChildren();
try{const body=new FormData();body.append('file',f.files[0]);const r=await fetch('/api/predict',{method:'POST',body});const data=await r.json();if(!r.ok)throw Error(data.error||'Request failed');
let description=data.mode==='pretrained'?'Pretrained chest X-ray model: pneumonia output. A low score is not a Normal diagnosis.':('Locally trained binary model. Validation accuracy: '+data.validation_accuracy_percent.toFixed(1)+'% ('+data.validation_count+' images). This is not clinical validation.');
document.getElementById('result').innerHTML='<strong>'+data.prediction+'</strong><br>Model score: '+data.score_percent.toFixed(1)+'%<br><small>'+description+'</small>';
if(data.mode==='pretrained' && data.findings){
  const box=document.getElementById('findings');
  const heading=document.createElement('h3');heading.textContent='Other chest X-ray findings';box.appendChild(heading);
  const note=document.createElement('p');note.textContent='Independent model scores, not confirmed diseases. They do not add up to 100%.';box.appendChild(note);
  const list=document.createElement('ul');list.className='finding-list';
  for(const item of data.findings){
    const row=document.createElement('li');const name=document.createElement('span');const value=document.createElement('strong');
    name.textContent=item.name;value.textContent=item.score_percent.toFixed(1)+'%';row.append(name,value);list.appendChild(row);
  }box.appendChild(list);
}
const heat=document.getElementById('heatmap');heat.src='data:image/png;base64,'+data.heatmap_png_base64;heat.style.display='block';document.getElementById('heatnote').textContent='Grad-CAM: warm colors show areas influencing this output; they do not prove disease location.';msg.textContent='Analysis complete.';
}catch(e){msg.textContent=e.message}finally{b.disabled=false}};
</script></body></html>'''


def torch_parts():
    try:
        import torch
        from torch import nn
        from torchvision import datasets, models, transforms
    except ImportError as exc:
        raise RuntimeError("Install dependencies first: pip install -r requirements.txt") from exc
    return torch, nn, datasets, models, transforms


def preprocessing(transforms, training=False):
    steps = [transforms.Resize((224, 224))]
    if training:
        steps += [transforms.RandomRotation(8), transforms.RandomHorizontalFlip()]
    steps += [transforms.ToTensor(), transforms.Normalize([.485, .456, .406], [.229, .224, .225])]
    return transforms.Compose(steps)


def heatmap_png(image, heat):
    import numpy as np
    base = image.resize((224, 224)).convert('RGBA')
    overlay = np.zeros((224, 224, 4), dtype=np.uint8)
    overlay[:, :, :3] = (255, 90, 20)
    overlay[:, :, 3] = np.uint8(np.clip(heat * 150, 0, 150))
    output = Image.alpha_composite(base, Image.fromarray(overlay, 'RGBA')).convert('RGB')
    buffer = io.BytesIO()
    output.save(buffer, format='PNG')
    return base64.b64encode(buffer.getvalue()).decode('ascii')


def valid_image(raw):
    try:
        image = Image.open(io.BytesIO(raw))
        image.load()
        if image.format not in ('JPEG', 'PNG'):
            raise ValueError('Only JPG and PNG files are supported.')
        return ImageOps.exif_transpose(image).convert('RGB')
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ValueError('Please upload a valid JPG or PNG chest X-ray.') from exc


def train(args):
    torch, nn, datasets, models, transforms = torch_parts()
    torch.manual_seed(42)
    train_dir, val_dir = ROOT / 'dataset/train', ROOT / 'dataset/val'
    if not train_dir.is_dir() or not val_dir.is_dir():
        raise RuntimeError('Create dataset/train and dataset/val with NORMAL and PNEUMONIA folders in each.')
    train_data = datasets.ImageFolder(train_dir, transform=preprocessing(transforms, True))
    val_data = datasets.ImageFolder(val_dir, transform=preprocessing(transforms))
    names = ['NORMAL', 'PNEUMONIA']
    if train_data.classes != names or val_data.classes != names:
        raise RuntimeError('Both splits must have exactly NORMAL and PNEUMONIA folders with images.')
    if len(train_data) < 2 or len(val_data) < 2:
        raise RuntimeError('Add labelled images to both training and validation splits.')
    # Pretrained ImageNet weights download once; --scratch works offline but needs more data.
    weights = None if args.scratch else models.ResNet18_Weights.DEFAULT
    try:
        model = models.resnet18(weights=weights)
    except Exception as exc:
        raise RuntimeError('Could not download ImageNet weights. Retry online or run train --scratch.') from exc
    if not args.scratch:
        for parameter in model.parameters():
            parameter.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, 2)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=0)
    validation = torch.utils.data.DataLoader(val_data, batch_size=args.batch_size, num_workers=0)
    optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=.001)
    loss_fn = nn.CrossEntropyLoss()
    best = -1
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(y)
        model.eval()
        correct = count = 0
        with torch.no_grad():
            for x, y in validation:
                predicted = model(x.to(device)).argmax(1).cpu()
                correct += int((predicted == y).sum())
                count += len(y)
        accuracy = correct / count
        print(f'Epoch {epoch+1}/{args.epochs}: train loss {total_loss/len(train_data):.4f}; validation accuracy {accuracy:.1%} ({count} images)', flush=True)
        if accuracy > best:
            best = accuracy
            torch.save({'state_dict': model.cpu().state_dict(), 'classes': names,
                        'validation_accuracy': best, 'validation_count': count,
                        'pretrained': not args.scratch}, MODEL_PATH)
            model.to(device)
    print(f'Saved best model to {MODEL_PATH}. Restart the server to load it.')


class Predictor:
    def __init__(self):
        self.model = None
        self.checkpoint = None
        self.lock = threading.Lock()
        if MODEL_PATH.exists():
            torch, nn, _, models, self.transforms = torch_parts()
            # Only load checkpoints you created or trust. weights_only prevents arbitrary pickle objects.
            checkpoint = torch.load(MODEL_PATH, map_location='cpu', weights_only=True)
            if checkpoint.get('classes') != ['NORMAL', 'PNEUMONIA']:
                raise RuntimeError('The saved model has unexpected class names.')
            model = models.resnet18(weights=None)
            model.fc = nn.Linear(model.fc.in_features, 2)
            model.load_state_dict(checkpoint['state_dict'])
            self.model = model.eval()
            self.checkpoint = checkpoint

    def predict(self, raw):
        torch, _, _, _, _ = torch_parts()
        image = valid_image(raw)
        tensor = preprocessing(self.transforms)(image).unsqueeze(0)
        activations, gradients = [], []
        # Hook the final convolution: BasicBlock's final ReLU is in-place and
        # cannot safely be used with a full backward hook on the block itself.
        layer = self.model.layer4[-1].conv2
        hook = layer.register_forward_hook(lambda module, inputs, output: activations.append(output))
        grad_hook = layer.register_full_backward_hook(lambda module, inputs, output: gradients.append(output[0]))
        try:
            self.model.zero_grad(set_to_none=True)
            logits = self.model(tensor)
            index = int(logits.argmax(1).item())
            score = float(logits.softmax(1)[0, index].item())
            logits[0, index].backward()
            weights = gradients[0].mean(dim=(2, 3), keepdim=True)
            heat = (weights * activations[0]).sum(1).relu()[0].detach()
            heat = heat - heat.min()
            heat = heat / heat.max().clamp(min=1e-8)
            heat = torch.nn.functional.interpolate(heat[None, None], size=(224, 224), mode='bilinear', align_corners=False)[0, 0].numpy()
        finally:
            hook.remove()
            grad_hook.remove()
        return {'mode': 'local', 'prediction': 'Model class: ' + self.checkpoint['classes'][index], 'score_percent': score * 100,
                'validation_accuracy_percent': self.checkpoint['validation_accuracy'] * 100,
                'validation_count': self.checkpoint['validation_count'],
                'heatmap_png_base64': heatmap_png(image, heat)}


class PretrainedPredictor:
    """TorchXRayVision all-datasets model; pneumonia is one of 18 outputs."""
    def __init__(self):
        try:
            import torchxrayvision as xrv
        except ImportError as exc:
            raise RuntimeError('Install dependencies first: pip install -r requirements.txt') from exc
        self.xrv = xrv
        self.lock = threading.Lock()
        try:
            self.model = xrv.models.DenseNet(weights='densenet121-res224-all',
                                             cache_dir=str(ROOT / 'weights')).eval()
        except Exception as exc:
            raise RuntimeError('Could not load pretrained chest X-ray weights. Check the weights folder or your internet connection.') from exc
        self.index = self.model.pathologies.index('Pneumonia')

    def predict(self, raw):
        import numpy as np
        import torch
        import torch.nn.functional as F
        image = valid_image(raw).convert('L')
        arr = self.xrv.datasets.normalize(np.asarray(image), 255)[None, ...]
        transform = __import__('torchvision').transforms.Compose([
            self.xrv.datasets.XRayCenterCrop(), self.xrv.datasets.XRayResizer(224)])
        tensor = torch.from_numpy(transform(arr)).unsqueeze(0)
        # Keep early layers outside the gradient graph to fit small CPU hosts.
        with torch.no_grad():
            early = tensor
            for name, layer in self.model.features.named_children():
                if name == 'denseblock4':
                    break
                early = layer(early)
        activation = self.model.features.denseblock4(early.detach())
        last = self.model.features.norm5(activation)
        pooled = F.adaptive_avg_pool2d(F.relu(last), (1, 1)).view(1, -1)
        logits = self.model.classifier(pooled)
        scores = self.xrv.models.op_norm(torch.sigmoid(logits), self.model.op_threshs)[0]
        value = float(scores[self.index].item())
        gradients = torch.autograd.grad(logits[0, self.index], activation)[0]
        weights = gradients.mean(dim=(2, 3), keepdim=True)
        heat = (weights * activation).sum(dim=1).relu()[0].detach()
        heat = (heat - heat.min()) / (heat.max() - heat.min()).clamp(min=1e-8)
        heat = F.interpolate(heat[None, None], size=(224, 224), mode='bilinear', align_corners=False)[0, 0].numpy()
        # These are separate pathology outputs, not a probability distribution.
        shown = ('Atelectasis', 'Consolidation', 'Effusion', 'Edema',
                 'Pneumothorax', 'Cardiomegaly', 'Lung Opacity', 'Fracture')
        values = dict(zip(self.model.pathologies, scores.detach().tolist()))
        findings = [
            {'name': name, 'score_percent': float(values[name]) * 100}
            for name in shown if name in values
        ]
        # Match the XRayCenterCrop input so the overlay aligns with the model.
        side = min(image.size)
        left = (image.width - side) // 2
        top = (image.height - side) // 2
        cropped = image.crop((left, top, left + side, top + side))
        return {'mode': 'pretrained', 'prediction': 'Pneumonia-related model output',
                'score_percent': value * 100, 'findings': findings,
                'heatmap_png_base64': heatmap_png(cropped, heat)}


def make_handler(predictor):
    class Handler(BaseHTTPRequestHandler):
        def send(self, status, body, content_type):
            self.send_response(status)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.end_headers()
            self.wfile.write(body)

        def json(self, status, data):
            self.send(status, json.dumps(data).encode('utf-8'), 'application/json; charset=utf-8')

        def do_GET(self):
            if self.path == '/':
                self.send(200, PAGE.encode('utf-8'), 'text/html; charset=utf-8')
            elif self.path == '/api/health':
                self.json(200, {'status': 'running', 'model_loaded': predictor.model is not None,
                                'mode': 'pretrained' if isinstance(predictor, PretrainedPredictor) else 'local'})
            else:
                self.json(404, {'error': 'Not found'})

        def do_POST(self):
            if self.path != '/api/predict':
                return self.json(404, {'error': 'Not found'})
            if predictor.model is None:
                return self.json(503, {'error': 'No model loaded. Check the weights folder or train on labelled X-rays and restart.'})
            try:
                length = int(self.headers.get('Content-Length', '0'))
                if not 0 < length <= MAX_UPLOAD + 65536:
                    return self.json(413, {'error': 'Image must be under 10 MB.'})
                content_type = self.headers.get('Content-Type', '')
                if not content_type.startswith('multipart/form-data;'):
                    return self.json(400, {'error': 'Expected an image upload.'})
                message = BytesParser(policy=default).parsebytes(b'Content-Type: ' + content_type.encode('ascii') + b'\r\nMIME-Version: 1.0\r\n\r\n' + self.rfile.read(length))
                parts = [p for p in message.iter_parts() if p.get_param('name', header='content-disposition') == 'file']
                if len(parts) != 1:
                    raise ValueError('Upload one file in the file field.')
                raw = parts[0].get_payload(decode=True)
                if not raw or len(raw) > MAX_UPLOAD:
                    raise ValueError('Image must be under 10 MB.')
                # Hooks are installed per inference on one shared model.
                with predictor.lock:
                    result = predictor.predict(raw)
                self.json(200, result)
            except (ValueError, UnicodeError) as exc:
                self.json(400, {'error': str(exc)})

    return Handler


def main():
    if not (3, 10) <= sys.version_info[:2] <= (3, 12):
        raise RuntimeError('Use Python 3.12 for this project. On Windows run start_windows.cmd, which selects it automatically.')
    parser = argparse.ArgumentParser(description='One-host chest X-ray educational project')
    commands = parser.add_subparsers(dest='command')
    serve = commands.add_parser('serve', help='start web app')
    serve.add_argument('--port', type=int, default=8000)
    serve.add_argument('--host', default='127.0.0.1', help='use 0.0.0.0 for other devices on your network')
    serve.add_argument('--pretrained', action='store_true', help='use published model even when a locally trained model exists')
    train_cmd = commands.add_parser('train', help='train a model using your labelled dataset')
    train_cmd.add_argument('--epochs', type=int, default=5)
    train_cmd.add_argument('--batch-size', type=int, default=16)
    train_cmd.add_argument('--scratch', action='store_true', help='offline random backbone; lower accuracy expected')
    args = parser.parse_args()
    if args.command == 'train':
        if args.epochs < 1 or args.batch_size < 1:
            parser.error('epochs and batch size must be positive')
        train(args)
    else:
        if args.command is None:
            args.port, args.host, args.pretrained = 8000, '127.0.0.1', False
        predictor = PretrainedPredictor() if args.pretrained or not MODEL_PATH.exists() else Predictor()
        server = ThreadingHTTPServer((args.host, args.port), make_handler(predictor))
        print(f'Open http://127.0.0.1:{args.port} — model {"ready" if predictor.model else "not loaded"}', flush=True)
        server.serve_forever()


if __name__ == '__main__':
    try:
        main()
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc
