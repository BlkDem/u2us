// Browser/Node integration contract. The ONNX graph is a decoder forward pass;
// an application should call it autoregressively until EOS.
export async function translate(text, source = 'ru', target = 'en') {
  const response = await fetch('/translate', {
    method: 'POST', headers: {'content-type': 'application/json'},
    body: JSON.stringify({text, source, target})
  });
  if (!response.ok) throw new Error(await response.text());
  return response.json();
}
