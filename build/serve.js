const http = require('http'), fs = require('fs'), path = require('path');
const ROOT = 'D:\\ragproj';
const TYPES = { '.html':'text/html; charset=utf-8', '.pdf':'application/pdf',
                '.js':'text/javascript', '.png':'image/png', '.css':'text/css' };
http.createServer((req, res) => {
  const rel = decodeURIComponent(req.url.split('?')[0]).replace(/^\/+/, '');
  const file = path.join(ROOT, rel);
  if (!file.startsWith(ROOT)) { res.writeHead(403); return res.end(); }
  fs.readFile(file, (err, buf) => {
    if (err) { res.writeHead(404); return res.end('404'); }
    res.writeHead(200, { 'Content-Type': TYPES[path.extname(file).toLowerCase()] || 'application/octet-stream' });
    res.end(buf);
  });
}).listen(8765, '127.0.0.1', () => console.log('serving', ROOT, 'on http://127.0.0.1:8765'));
