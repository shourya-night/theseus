/**
 * Renders scripts/shaderPreview.ts to /tmp/shaders.png in headless Chromium.
 * See the header of shaderPreview.ts for usage. The executablePath below points
 * at a preinstalled Chromium; drop it to let Playwright use its own.
 */
import { chromium } from 'playwright';
const b = await chromium.launch({ executablePath: '/opt/pw-browsers/chromium-1194/chrome-linux/chrome', args: ['--no-sandbox','--use-gl=angle','--use-angle=swiftshader','--enable-unsafe-swiftshader'] });
const p = await b.newPage({ viewport: { width: 1700, height: 440 }, deviceScaleFactor: 2 });
const errs = [];
p.on('console', m => { if (m.type() === 'error' || m.type() === 'warning') errs.push(m.text()); });
p.on('pageerror', e => errs.push('PAGEERROR: ' + e.message));
await p.goto('file:///tmp/preview.html');
await p.waitForFunction('window.__done === true', { timeout: 20000 }).catch(() => errs.push('TIMEOUT: __done never set'));
await p.waitForTimeout(600);
await p.screenshot({ path: '/tmp/shaders.png' });
console.log(errs.length ? errs.join('\n---\n') : 'no console errors');
await b.close();
