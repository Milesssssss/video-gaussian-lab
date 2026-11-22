import { readFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import { expect, test, type Page } from '@playwright/test';

async function visibleSceneRatio(page: Page, screenshot: Buffer): Promise<number> {
  return page.evaluate(async (encoded) => {
    const bytes = Uint8Array.from(atob(encoded), (character) => character.charCodeAt(0));
    const image = await createImageBitmap(new Blob([bytes], { type: 'image/png' }));
    const sample = document.createElement('canvas');
    sample.width = 128;
    sample.height = 96;
    const context = sample.getContext('2d')!;
    // Sample the actual compositor screenshot, excluding toolbar/HUD/timeline overlays.
    // Reading WebGL directly after presentation can return a cleared drawing buffer.
    context.drawImage(image, image.width * 0.15, image.height * 0.18,
      image.width * 0.7, image.height * 0.5, 0, 0, sample.width, sample.height);
    image.close();
    const pixels = context.getImageData(0, 0, sample.width, sample.height).data;
    let visible = 0;
    for (let offset = 0; offset < pixels.length; offset += 4) {
      if (Math.max(pixels[offset], pixels[offset + 1], pixels[offset + 2]) > 80) visible++;
    }
    return visible / (sample.width * sample.height);
  }, screenshot.toString('base64'));
}

test('synthetic workspace responds to controls and exports explicitly marked point clouds', async ({ page }, testInfo) => {
  const errors: string[] = [];
  page.on('pageerror', (error) => errors.push(error.message));
  page.on('console', (message) => {
    if (message.type() === 'error' && /WebGL|Shader|VALIDATE_STATUS/i.test(message.text())) {
      errors.push(message.text());
    }
  });
  await page.goto('/');
  await expect(page.locator('.source-badge').first()).toHaveText('合成演示');
  const viewport = page.getByTestId('gaussian-webgl-viewport');
  await expect(viewport.locator('canvas')).toBeVisible();
  await expect(viewport).toHaveAttribute('data-render-state', 'ready');
  const renderedCanvas = await viewport.screenshot({ path: testInfo.outputPath('webgl-canvas.png') });
  const ratio = await visibleSceneRatio(page, renderedCanvas);
  console.info(`${testInfo.project.name} visible scene pixel ratio: ${ratio.toFixed(4)}`);
  expect(ratio, 'The center of the 3D viewport must contain visible scene pixels, not just an empty canvas').toBeGreaterThan(0.01);
  await expect(page.getByRole('alert')).toHaveCount(0);
  await page.screenshot({ path: testInfo.outputPath('workbench.png') });
  if (testInfo.project.name === 'mobile') await page.getByRole('button', { name: '切换参数面板' }).click();
  await page.getByRole('button', { name: '实验桌面推轨', exact: true }).click();
  await expect(page.locator('.app-shell')).toHaveAttribute('data-sequence-id', 'workbench');
  await expect(viewport).toHaveAttribute('data-render-state', 'ready');
  await page.getByRole('button', { name: '逐帧', exact: true }).click();
  await expect(page.getByRole('slider', { name: '深度校正权重', exact: true })).toBeDisabled();
  await page.getByRole('button', { name: '聚合', exact: true }).click();
  await expect(page.getByRole('slider', { name: '深度校正权重', exact: true })).toBeEnabled();
  const downloadEvent = page.waitForEvent('download');
  await page.getByRole('button', { name: '导出演示点云', exact: true }).click();
  const download = await downloadEvent;
  expect(download.suggestedFilename()).toBe('workbench.synthetic-pointcloud.ply');
  const output = await readFile((await download.path())!, 'utf8');
  expect(output).toContain('synthetic_demo_not_a_reconstruction_result');
  expect(output).not.toContain('property float rot_');
  if (testInfo.project.name === 'mobile') await page.getByRole('button', { name: '切换参数面板' }).click();
  await page.getByRole('button', { name: '跳转到第 4 帧', exact: true }).click();
  await expect(page.getByRole('button', { name: '跳转到第 4 帧', exact: true })).toHaveAttribute('aria-pressed', 'true');
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(true);
  expect(errors).toEqual([]);
});

test('research status has six arms and no fabricated results; inspector supports keyboard probing', async ({ page }, testInfo) => {
  const errors: string[] = [];
  page.on('pageerror', (error) => errors.push(error.message));
  await page.goto('/');
  await page.getByRole('button', { name: '实验对照', exact: true }).click();
  await expect(page.getByRole('heading', { name: '实验对照与就绪状态' })).toBeVisible();
  await expect(page.locator('tbody tr')).toHaveCount(6);
  await expect(page.getByTestId('research-results-state')).toHaveText('尚未提供实验结果');
  await expect(page.getByTestId('protocol-status')).toHaveText('草案 · 未冻结');
  await page.screenshot({ path: testInfo.outputPath('experiments.png') });
  await page.getByRole('button', { name: '管线方案', exact: true }).click();
  await expect(page.getByRole('heading', { name: 'Phase 1 · 固定相机的共享高斯场景' })).toBeVisible();
  await page.getByRole('button', { name: '特征与深度', exact: true }).click();
  const probe = page.getByRole('group', { name: '合成深度探针，使用方向键移动' });
  await probe.focus();
  await probe.press('Home');
  await expect(page.getByText('(0, 0)', { exact: false })).toBeVisible();
  await probe.press('Shift+ArrowRight');
  await expect(page.getByText('(10, 0)', { exact: false })).toBeVisible();
  await probe.press('Escape');
  await expect(page.getByText('尚未选择像素', { exact: true })).toBeVisible();
  await page.screenshot({ path: testInfo.outputPath('inspector.png') });
  expect(errors).toEqual([]);
});

test('local video preview decodes six frames without replacing or uploading the synthetic scene', async ({ page }, testInfo) => {
  const errors: string[] = [];
  const external: string[] = [];
  page.on('pageerror', (error) => errors.push(error.message));
  page.on('request', (request) => { if (/^https?:/.test(request.url()) && new URL(request.url()).hostname !== '127.0.0.1') external.push(request.url()); });
  await page.goto('/');
  const button = page.getByRole('button', { name: '导入视频', exact: true });
  await button.click();
  const dialog = page.getByRole('dialog', { name: '本地视频预览' });
  await expect(dialog).toBeVisible();
  await dialog.locator('input[type=file]').setInputFiles({ name: 'not-video.txt', mimeType: 'text/plain', buffer: Buffer.from('invalid video') });
  await expect(dialog.getByRole('alert')).toContainText('请选择 MP4');
  await dialog.locator('input[type=file]').setInputFiles(fileURLToPath(new URL('./fixtures/local-preview.webm', import.meta.url)));
  await expect(dialog.getByRole('img')).toHaveCount(6);
  await expect(dialog.getByText('local-preview.webm', { exact: true })).toBeVisible();
  await expect(dialog.getByText('160 × 96', { exact: false })).toBeVisible();
  await page.screenshot({ path: testInfo.outputPath('video-preview.png') });
  await page.keyboard.press('Escape');
  await expect(dialog).not.toBeVisible();
  await expect(page.locator('.app-shell')).toHaveAttribute('data-sequence-id', 'courtyard');
  await expect(page.getByRole('button', { name: '查看视频', exact: true })).toBeFocused();
  expect(external).toEqual([]);
  expect(errors).toEqual([]);
});
