import { useEffect, useLayoutEffect, useRef, useState, type FC } from 'react';
import {
  ACESFilmicToneMapping,
  AxesHelper,
  BoxGeometry,
  BufferAttribute,
  BufferGeometry,
  Color,
  ConeGeometry,
  CylinderGeometry,
  DataTexture,
  DirectionalLight,
  DoubleSide,
  EdgesGeometry,
  Float32BufferAttribute,
  FogExp2,
  GridHelper,
  Group,
  HemisphereLight,
  InstancedBufferAttribute,
  InstancedBufferGeometry,
  InstancedMesh,
  Line,
  LineBasicMaterial,
  LineLoop,
  LineSegments,
  LinearFilter,
  type Material,
  MathUtils,
  Matrix4,
  Mesh,
  MeshBasicMaterial,
  MeshStandardMaterial,
  OctahedronGeometry,
  PMREMGenerator,
  PerspectiveCamera,
  PlaneGeometry,
  Quaternion,
  RGBAFormat,
  SRGBColorSpace,
  Scene,
  ShaderMaterial,
  SphereGeometry,
  type Texture,
  Vector3,
  WebGLRenderer,
  type WebGLRenderTarget,
} from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';
import type { AggregationResult, CameraPose, CameraPreset, ColorMode } from '../types/pipeline';
import {
  computeSplatDepthConfidence,
  confidenceToRgb01,
  type DepthConfidenceResult,
} from '../demo/depthConfidence';

interface GaussianViewportProps {
  result: AggregationResult;
  gtPoses: CameraPose[];
  rawPoses: CameraPose[];
  activeFrameIndex: number;
  colorMode: ColorMode;
  layers: {
    showSplats: boolean;
    showCameras: boolean;
    showTracks: boolean;
    showFeatureLinks: boolean;
    showRawGhost: boolean;
    showDepthConfidence: boolean;
  };
  depthConfidence: DepthConfidenceResult;
  depthTemporalWeight: number;
}

type RenderState = 'pending' | 'ready' | 'lost' | 'error';

function depthToColor(depth: number, out: Color) {
  const t = 1 - Math.max(0, Math.min(1, (depth - 1.2) / 6.5));
  const r = t > 0.55 ? (t - 0.55) * 2.2 : 0.05;
  const g = 0.18 + 0.76 * Math.sin(t * Math.PI * 0.85);
  const b = 0.28 + 0.68 * (1 - Math.abs(t - 0.45) * 1.6);
  out.setRGB(
    Math.max(0, Math.min(1, r)),
    Math.max(0, Math.min(1, g)),
    Math.max(0, Math.min(1, b)),
  );
}

function errorToColor(errMeters: number, out: Color) {
  const t = Math.max(0, Math.min(1, errMeters / 0.22));
  // Cyan (low residual) -> Slate -> Amber/Rose (high drift residual)
  if (t < 0.45) {
    const k = t / 0.45;
    out.setRGB(0.03 + k * 0.2, 0.71 - k * 0.1, 0.83 - k * 0.3);
  } else {
    const k = (t - 0.45) / 0.55;
    out.setRGB(0.23 + k * 0.73, 0.61 - k * 0.25, 0.53 - k * 0.43);
  }
}

export const GaussianViewport: FC<GaussianViewportProps> = ({
  result,
  rawPoses,
  activeFrameIndex,
  colorMode,
  layers,
  depthConfidence,
  depthTemporalWeight,
}) => {
  const mountRef = useRef<HTMLDivElement | null>(null);
  const hudCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const [contextLost, setContextLost] = useState(false);
  const [renderError, setRenderError] = useState<string | null>(null);
  const [renderState, setRenderState] = useState<RenderState>('pending');
  const [hasPresentedFrame, setHasPresentedFrame] = useState(false);
  const [retryKey, setRetryKey] = useState(0);
  const [activePreset, setActivePreset] = useState<CameraPreset>('free');

  const syncSceneRef = useRef<() => void>(() => {});
  const applyCameraPresetRef = useRef<(preset: CameraPreset) => void>(() => {});

  const latestPropsRef = useRef({
    result,
    rawPoses,
    activeFrameIndex,
    colorMode,
    layers,
    depthConfidence,
    depthTemporalWeight,
  });
  latestPropsRef.current = {
    result,
    rawPoses,
    activeFrameIndex,
    colorMode,
    layers,
    depthConfidence,
    depthTemporalWeight,
  };

  useEffect(() => {
    const container = mountRef.current;
    if (!container) return;
    setContextLost(false);
    setRenderError(null);
    setHasPresentedFrame(false);

    let disposed = false;
    let contextUnavailable = false;
    let renderFailed = false;
    let frameId = 0;
    let timerId = 0;
    let pendingSync: WebGLSync | null = null;

    function publishRenderState(state: RenderState) {
      if (disposed || !container) return;
      // Update synchronously so a screenshot cannot observe an old ready state after invalidation.
      container.dataset.renderState = state;
      setRenderState(state);
    }
    publishRenderState('pending');

    const _vecA = new Vector3();
    const _vecB = new Vector3();
    const _targetPos = new Vector3();
    const _targetLook = new Vector3();
    const _mat4 = new Matrix4();
    const _quatIdentity = new Quaternion();
    const _scaleVec = new Vector3();
    const _anchorColor = new Color();

    let isSoftwareGl = true;
    let renderer: WebGLRenderer;
    try {
      const probe = document.createElement('canvas').getContext('webgl2');
      if (probe) {
        const ext = probe.getExtension('WEBGL_debug_renderer_info');
        const param = ext ? ext.UNMASKED_RENDERER_WEBGL : probe.RENDERER;
        const gpu = String(probe.getParameter(param));
        isSoftwareGl = /SwiftShader|llvmpipe|softpipe|Software|Basic Render/i.test(gpu);
        probe.getExtension('WEBGL_lose_context')?.loseContext();
      }
      renderer = new WebGLRenderer({
        antialias: !isSoftwareGl,
        powerPreference: 'high-performance',
      });
    } catch {
      publishRenderState('error');
      setRenderError('当前浏览器无法创建 WebGL 视图。可检查硬件加速设置后重试，其他面板仍可使用。');
      return;
    }
    renderer.outputColorSpace = SRGBColorSpace;
    renderer.toneMapping = ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.05;
    renderer.setPixelRatio(isSoftwareGl ? 1 : Math.min(window.devicePixelRatio, 2));
    renderer.shadowMap.enabled = false;

    const width = Math.max(1, container.clientWidth);
    const height = Math.max(1, container.clientHeight);
    let renderedWidth = width;
    let renderedHeight = height;
    renderer.setSize(width, height, false);
    renderer.domElement.style.display = 'block';
    renderer.domElement.style.width = '100%';
    renderer.domElement.style.height = '100%';
    container.appendChild(renderer.domElement);

    const onContextLost = (e: Event) => {
      e.preventDefault();
      contextUnavailable = true;
      cancelAnimationFrame(frameId);
      window.clearTimeout(timerId);
      pendingSync = null; // Context loss invalidates the GPU sync object.
      renderBusy = false;
      publishRenderState('lost');
      setContextLost(true);
    };
    const onContextRestored = () => {
      publishRenderState('pending');
      setContextLost(false);
      // Recreate offscreen environment targets and event handlers after context restoration.
      setRetryKey((value) => value + 1);
    };
    renderer.domElement.addEventListener('webglcontextlost', onContextLost);
    renderer.domElement.addEventListener('webglcontextrestored', onContextRestored);

    const scene = new Scene();
    scene.background = new Color(0x07090e);
    scene.fog = new FogExp2(0x07090e, 0.036);

    const pmremGenerator = new PMREMGenerator(renderer);
    const roomEnvironment = new RoomEnvironment();
    let environmentTarget: WebGLRenderTarget | null = null;
    try {
      environmentTarget = pmremGenerator.fromScene(roomEnvironment, 0.04);
      scene.environment = environmentTarget.texture;
    } catch {
      // The environment is decorative; the explicit lights remain usable without it.
      scene.environment = null;
    } finally {
      roomEnvironment.dispose();
      pmremGenerator.dispose();
    }

    const camera = new PerspectiveCamera(48, width / height, 0.1, 100);
    camera.position.set(5.4, 3.8, 5.6);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.target.set(0, 0.9, 0);
    controls.enableDamping = !isSoftwareGl;
    controls.dampingFactor = 0.08;
    controls.minDistance = 1.6;
    controls.maxDistance = 18;
    controls.maxPolarAngle = Math.PI * 0.49;

    // 3-point warm/cool lighting for camera frustum housings
    const hemiLight = new HemisphereLight(0xcbd5e1, 0x0f172a, 0.85);
    scene.add(hemiLight);
    const keyLight = new DirectionalLight(0xfff1e0, 1.4);
    keyLight.position.set(6, 9, 5);
    scene.add(keyLight);
    const rimLight = new DirectionalLight(0x38bdf8, 0.65);
    rimLight.position.set(-6, 4, -5);
    scene.add(rimLight);

    // Synthetic scene reference grid (offset below ground splats to avoid z-fighting)
    const gridHelper = new GridHelper(12, 24, 0x1e293b, 0x111827);
    gridHelper.position.y = -0.06;
    scene.add(gridHelper);

    const axesHelper = new AxesHelper(0.8);
    axesHelper.position.set(-4.5, -0.04, -4.5);
    scene.add(axesHelper);

    // Instanced camera-facing Gaussian billboard preview (not a full 3DGS renderer)
    const baseQuad = new PlaneGeometry(2, 2);
    const instancedGeo = new InstancedBufferGeometry();
    instancedGeo.index = baseQuad.index;
    instancedGeo.attributes.position = baseQuad.attributes.position;

    const MAX_SPLATS = 16000;
    const offsetAttr = new InstancedBufferAttribute(new Float32Array(MAX_SPLATS * 3), 3);
    const scaleAttr = new InstancedBufferAttribute(new Float32Array(MAX_SPLATS * 2), 2);
    const colorAttr = new InstancedBufferAttribute(new Float32Array(MAX_SPLATS * 4), 4);
    instancedGeo.setAttribute('aOffset', offsetAttr);
    instancedGeo.setAttribute('aScale', scaleAttr);
    instancedGeo.setAttribute('aColor', colorAttr);
    instancedGeo.instanceCount = 0;

    const splatMaterial = new ShaderMaterial({
      uniforms: {
        uSplatGain: { value: 1.38 },
      },
      vertexShader: `
        attribute vec3 aOffset;
        attribute vec2 aScale;
        attribute vec4 aColor;
        uniform float uSplatGain;
        varying vec2 vQuadUv;
        varying vec4 vColor;

        void main() {
          vQuadUv = position.xy;
          vColor = aColor;
          vec4 mvCenter = viewMatrix * vec4(aOffset, 1.0);
          mvCenter.xy += position.xy * aScale * uSplatGain;
          gl_Position = projectionMatrix * mvCenter;
        }
      `,
      fragmentShader: `
        varying vec2 vQuadUv;
        varying vec4 vColor;

        void main() {
          float r2 = dot(vQuadUv, vQuadUv);
          if (r2 > 1.0) discard;
          float gaussian = exp(-2.8 * r2);
          float alpha = vColor.a * gaussian;
          if (alpha < 0.02) discard;
          gl_FragColor = vec4(vColor.rgb, alpha);
        }
      `,
      transparent: true,
      depthWrite: false,
      depthTest: true,
    });

    const splatMesh = new Mesh(instancedGeo, splatMaterial);
    splatMesh.frustumCulled = false;
    splatMesh.renderOrder = 2;
    scene.add(splatMesh);

    // Camera Trajectory & Frustums Group
    const cameraGroup = new Group();
    scene.add(cameraGroup);

    // Feature Correspondence Rays Group (camera-to-landmark)
    const tracksGroup = new Group();
    scene.add(tracksGroup);

    // Consecutive-Frame Matched Global Feature Anchor Links Group
    const featureLinksGroup = new Group();
    featureLinksGroup.renderOrder = 4;
    scene.add(featureLinksGroup);

    // Per-Frame Synthetic Heuristic Heatmap Frustum Overlay Group
    const confidenceOverlayGroup = new Group();
    confidenceOverlayGroup.renderOrder = 3;
    scene.add(confidenceOverlayGroup);

    function disposeConfidenceOverlayGroup() {
      while (confidenceOverlayGroup.children.length > 0) {
        const ch = confidenceOverlayGroup.children[0];
        confidenceOverlayGroup.remove(ch);
        ch.traverse((obj) => {
          if (obj instanceof Mesh) {
            obj.geometry.dispose();
            const mats = Array.isArray(obj.material) ? obj.material : [obj.material];
            for (const m of mats) {
              if ('map' in m && m.map && typeof (m.map as Texture).dispose === 'function') {
                (m.map as Texture).dispose();
              }
              m.dispose();
            }
          } else if (obj instanceof LineSegments || obj instanceof LineLoop || obj instanceof Line) {
            obj.geometry.dispose();
            const mats = Array.isArray(obj.material) ? obj.material : [obj.material];
            for (const m of mats) m.dispose();
          }
        });
      }
    }

    // Shared geometries/materials for global feature anchor diamonds & endpoint nodes
    const anchorOctaGeo = new OctahedronGeometry(0.068, 0);
    const anchorOctaMat = new MeshStandardMaterial({
      color: 0xffffff,
      roughness: 0.25,
      metalness: 0.35,
      emissive: 0x064e3b,
      emissiveIntensity: 0.45,
    });
    const endpointNodeGeo = new SphereGeometry(0.026, 8, 8);
    const endpointNodeMat = new MeshBasicMaterial({
      color: 0xffffff,
      transparent: true,
      opacity: 0.9,
    });

    // Shared composite camera frustum geometries/materials
    const bodyGeo = new BoxGeometry(0.14, 0.10, 0.16);
    const lensGeo = new CylinderGeometry(0.045, 0.052, 0.07, 12);
    lensGeo.rotateX(Math.PI / 2);
    const frustumPyramidGeo = new ConeGeometry(0.18, 0.28, 4, 1, true);
    frustumPyramidGeo.rotateY(Math.PI / 4);
    frustumPyramidGeo.rotateX(-Math.PI / 2);
    const frustumEdgesGeo = new EdgesGeometry(frustumPyramidGeo);

    const inactiveCamMat = new MeshStandardMaterial({
      color: 0x334155,
      roughness: 0.4,
      metalness: 0.7,
    });
    const activeCamMat = new MeshStandardMaterial({
      color: 0x06b6d4,
      roughness: 0.25,
      metalness: 0.8,
      emissive: 0x083344,
    });
    const lensMat = new MeshStandardMaterial({
      color: 0x0f172a,
      roughness: 0.15,
      metalness: 0.9,
    });
    const activeLineMat = new LineBasicMaterial({ color: 0x22d3ee });
    const inactiveLineMat = new LineBasicMaterial({ color: 0x475569 });
    const trajLineMat = new LineBasicMaterial({ color: 0x06b6d4 });
    const rawTrajLineMat = new LineBasicMaterial({
      color: 0xf59e0b,
      transparent: true,
      opacity: 0.55,
    });

    // Reusable render scheduling (software-WebGL safe)
    let renderBusy = false;
    let renderDirty = false;
    let cameraTweenActive = false;
    let lastClock = performance.now();

    function afterDraw(cb: () => void) {
      if (disposed || contextUnavailable) return;
      const gl = renderer.getContext() as WebGL2RenderingContext;
      pendingSync = gl.fenceSync(gl.SYNC_GPU_COMMANDS_COMPLETE, 0);
      if (!pendingSync) {
        renderBusy = false;
        renderFailed = true;
        publishRenderState('error');
        setRenderError('浏览器未能确认三维绘制完成，请重试视图。');
        return;
      }
      const fenceStartedAt = performance.now();
      gl.flush();
      const poll = () => {
        if (disposed || contextUnavailable) return;
        const state = pendingSync && gl.getSyncParameter(pendingSync, gl.SYNC_STATUS);
        if (state !== gl.SIGNALED) {
          if (performance.now() - fenceStartedAt > 15000) {
            if (pendingSync) gl.deleteSync(pendingSync);
            pendingSync = null;
            renderBusy = false;
            renderFailed = true;
            publishRenderState('error');
            setRenderError('三维绘制等待超时，请重试视图或检查浏览器图形加速。');
            return;
          }
          timerId = window.setTimeout(poll, 5);
        } else {
          if (pendingSync) gl.deleteSync(pendingSync);
          pendingSync = null;
          timerId = window.setTimeout(() => {
            if (disposed || contextUnavailable) return;
            // A visible canvas is not a presented frame. Let the browser present the completed GPU work.
            frameId = requestAnimationFrame(() => {
              if (!disposed && !contextUnavailable) cb();
            });
          }, isSoftwareGl ? 70 : 0);
        }
      };
      timerId = window.setTimeout(poll, 5);
    }

    function requestRender() {
      if (disposed || contextUnavailable || renderFailed) return;
      publishRenderState('pending');
      if (renderBusy) {
        renderDirty = true;
        return;
      }
      renderBusy = true;
      frameId = requestAnimationFrame(() => {
        if (disposed || contextUnavailable || renderFailed) return;
        renderDirty = false;
        const now = performance.now();
        const delta = Math.min(0.1, (now - lastClock) / 1000);
        lastClock = now;

        if (cameraTweenActive) {
          const dampK = 7.5;
          camera.position.x = MathUtils.damp(camera.position.x, _targetPos.x, dampK, delta);
          camera.position.y = MathUtils.damp(camera.position.y, _targetPos.y, dampK, delta);
          camera.position.z = MathUtils.damp(camera.position.z, _targetPos.z, dampK, delta);
          controls.target.x = MathUtils.damp(controls.target.x, _targetLook.x, dampK, delta);
          controls.target.y = MathUtils.damp(controls.target.y, _targetLook.y, dampK, delta);
          controls.target.z = MathUtils.damp(controls.target.z, _targetLook.z, dampK, delta);

          if (
            camera.position.distanceToSquared(_targetPos) < 0.0005 &&
            controls.target.distanceToSquared(_targetLook) < 0.0005
          ) {
            camera.position.copy(_targetPos);
            controls.target.copy(_targetLook);
            cameraTweenActive = false;
          } else {
            renderDirty = true;
          }
        }

        controls.update();
        try {
          renderer.render(scene, camera);
        } catch {
          renderBusy = false;
          renderFailed = true;
          publishRenderState('error');
          setRenderError('三维预览暂时无法绘制。可重试初始化视图，其他面板仍可使用。');
          return;
        }

        const done = () => {
          renderBusy = false;
          if (renderDirty) {
            requestRender();
          } else {
            setHasPresentedFrame(true);
            publishRenderState('ready');
          }
        };
        // Preserve hardware interaction cadence; only a stable frame can publish ready.
        if (isSoftwareGl || !renderDirty) afterDraw(done);
        else done();
      });
    }

    const tmpColor = new Color();
    const sortEntries: { idx: number; distSq: number }[] = [];

    function syncSceneData() {
      if (disposed || contextUnavailable || renderFailed) return;
      publishRenderState('pending');
      const {
        result: res,
        rawPoses: rPoses,
        activeFrameIndex: activeIdx,
        colorMode: cMode,
        layers: lyr,
        depthConfidence: confData,
        depthTemporalWeight: dpWeight,
      } = latestPropsRef.current;

      // 1. Update Gaussian Splats sorted back-to-front relative to camera
      splatMesh.visible = lyr.showSplats;
      if (lyr.showSplats) {
        const splats = res.splats;
        const count = Math.min(splats.length, MAX_SPLATS);
        sortEntries.length = count;
        const cx = camera.position.x;
        const cy = camera.position.y;
        const cz = camera.position.z;

        for (let i = 0; i < count; i++) {
          const p = splats[i].position;
          const dx = p[0] - cx;
          const dy = p[1] - cy;
          const dz = p[2] - cz;
          sortEntries[i] = { idx: i, distSq: dx * dx + dy * dy + dz * dz };
        }
        sortEntries.sort((a, b) => b.distSq - a.distSq);

        const offArr = offsetAttr.array as Float32Array;
        const sclArr = scaleAttr.array as Float32Array;
        const colArr = colorAttr.array as Float32Array;

        for (let i = 0; i < count; i++) {
          const sp = splats[sortEntries[i].idx];
          offArr[i * 3] = sp.position[0];
          offArr[i * 3 + 1] = sp.position[1];
          offArr[i * 3 + 2] = sp.position[2];

          sclArr[i * 2] = sp.scale[0];
          sclArr[i * 2 + 1] = sp.scale[1];

          if (cMode === 'rgb') {
            tmpColor.setRGB(sp.color[0], sp.color[1], sp.color[2]);
          } else if (cMode === 'depth') {
            depthToColor(sp.depthMeters, tmpColor);
          } else {
            errorToColor(sp.residualError, tmpColor);
          }

          if (lyr.showDepthConfidence) {
            const spConf = computeSplatDepthConfidence(
              sp.depthMeters,
              sp.residualError,
              dpWeight,
            );
            const [hr, hg, hb] = confidenceToRgb01(spConf);
            const blend = sp.frameIndex === activeIdx ? 0.88 : 0.64;
            tmpColor.setRGB(
              tmpColor.r * (1 - blend) + hr * blend,
              tmpColor.g * (1 - blend) + hg * blend,
              tmpColor.b * (1 - blend) + hb * blend,
            );
          }

          colArr[i * 4] = tmpColor.r;
          colArr[i * 4 + 1] = tmpColor.g;
          colArr[i * 4 + 2] = tmpColor.b;
          colArr[i * 4 + 3] = sp.opacity;
        }

        offsetAttr.needsUpdate = true;
        scaleAttr.needsUpdate = true;
        colorAttr.needsUpdate = true;
        instancedGeo.instanceCount = count;
      }

      // 2. Rebuild Camera Frustums & Trajectory
      while (cameraGroup.children.length > 0) {
        const ch = cameraGroup.children[0];
        cameraGroup.remove(ch);
        if (ch instanceof Line) ch.geometry.dispose();
      }
      cameraGroup.visible = lyr.showCameras;

      if (lyr.showCameras && res.poses.length > 0) {
        const trajPoints = res.poses.map((p) => new Vector3(...p.position));
        if (trajPoints.length > 1) {
          const trajGeo = new BufferGeometry().setFromPoints(trajPoints);
          cameraGroup.add(new Line(trajGeo, trajLineMat));
        }

        if (lyr.showRawGhost && rPoses.length > 1) {
          const slicedRaw = rPoses
            .slice(0, res.poses.length)
            .map((p) => new Vector3(...p.position));
          const rawGeo = new BufferGeometry().setFromPoints(slicedRaw);
          cameraGroup.add(new Line(rawGeo, rawTrajLineMat));
        }

        for (const pose of res.poses) {
          const isCurrent = pose.frameIndex === activeIdx;
          const camRig = new Group();
          camRig.position.set(...pose.position);
          _vecA.set(...pose.lookAt);
          camRig.lookAt(_vecA);

          const body = new Mesh(bodyGeo, isCurrent ? activeCamMat : inactiveCamMat);
          const lens = new Mesh(lensGeo, lensMat);
          lens.position.z = 0.1;
          const frustum = new LineSegments(
            frustumEdgesGeo,
            isCurrent ? activeLineMat : inactiveLineMat,
          );
          frustum.position.z = 0.24;

          camRig.add(body, lens, frustum);
          if (isCurrent) camRig.scale.setScalar(1.22);
          cameraGroup.add(camRig);
        }
      }

      // 3. Rebuild Global Feature Correspondence Rays
      while (tracksGroup.children.length > 0) {
        const ch = tracksGroup.children[0];
        tracksGroup.remove(ch);
        if (ch instanceof LineSegments) {
          ch.geometry.dispose();
          (ch.material as Material).dispose();
        }
      }
      tracksGroup.visible = lyr.showTracks;

      if (lyr.showTracks && res.featureTracks.length > 0) {
        const inlierPts: number[] = [];
        const outlierPts: number[] = [];
        const step = res.featureTracks.length > 120 ? 2 : 1;

        for (let i = 0; i < res.featureTracks.length; i += step) {
          const tr = res.featureTracks[i];
          const arr = tr.inlier ? inlierPts : outlierPts;
          arr.push(tr.from[0], tr.from[1], tr.from[2], tr.to[0], tr.to[1], tr.to[2]);
        }

        if (inlierPts.length > 0) {
          const g = new BufferGeometry();
          g.setAttribute('position', new Float32BufferAttribute(inlierPts, 3));
          const m = new LineBasicMaterial({
            color: 0x06b6d4,
            transparent: true,
            opacity: 0.16,
          });
          tracksGroup.add(new LineSegments(g, m));
        }

        if (outlierPts.length > 0) {
          const g = new BufferGeometry();
          g.setAttribute('position', new Float32BufferAttribute(outlierPts, 3));
          const m = new LineBasicMaterial({
            color: 0xf59e0b,
            transparent: true,
            opacity: 0.35,
          });
          tracksGroup.add(new LineSegments(g, m));
        }
      }

      // 4. Rebuild Consecutive-Frame Matched Global Feature Links & Anchors
      while (featureLinksGroup.children.length > 0) {
        const ch = featureLinksGroup.children[0];
        featureLinksGroup.remove(ch);
        if (ch instanceof LineSegments) {
          ch.geometry.dispose();
          (ch.material as Material).dispose();
        } else if (ch instanceof InstancedMesh) {
          ch.dispose();
        }
      }
      featureLinksGroup.visible = lyr.showFeatureLinks;

      if (lyr.showFeatureLinks && res.consecutiveFeatureLinks.length > 0) {
        const links = res.consecutiveFeatureLinks;
        const linkPos: number[] = [];
        const linkCol: number[] = [];
        const pullPos: number[] = [];
        const pullCol: number[] = [];

        for (let i = 0; i < links.length; i++) {
          const lk = links[i];
          // Consecutive-frame correspondence line (frame t -> frame t+1)
          linkPos.push(lk.from[0], lk.from[1], lk.from[2], lk.to[0], lk.to[1], lk.to[2]);

          if (!lk.inlier) {
            // Rose for rejected outliers
            _anchorColor.setRGB(0.95, 0.25, 0.37);
          } else {
            // Emerald (tightly anchored < 5cm) -> Cyan -> Amber (drifting >= 24cm)
            const t = Math.max(0, Math.min(1, (lk.gapMeters - 0.04) / 0.20));
            if (t < 0.4) {
              const k = t / 0.4;
              _anchorColor.setRGB(0.06 + k * 0.08, 0.85 - k * 0.06, 0.56 + k * 0.32);
            } else {
              const k = (t - 0.4) / 0.6;
              _anchorColor.setRGB(0.14 + k * 0.82, 0.79 - k * 0.17, 0.88 - k * 0.82);
            }
          }
          linkCol.push(
            _anchorColor.r,
            _anchorColor.g,
            _anchorColor.b,
            _anchorColor.r,
            _anchorColor.g,
            _anchorColor.b,
          );

          // Constraint pull-in tethers from unaligned raw synthetic observation -> anchored position
          if (lk.inlier && lk.pullMeters > 0.025) {
            pullPos.push(
              lk.rawFrom[0],
              lk.rawFrom[1],
              lk.rawFrom[2],
              lk.from[0],
              lk.from[1],
              lk.from[2],
            );
            // Amber tail (raw drift) -> Emerald head (anchored lock)
            pullCol.push(0.96, 0.62, 0.04, 0.06, 0.85, 0.56);
          }
        }

        // Primary consecutive-frame matched feature link segments
        if (linkPos.length > 0) {
          const linkGeo = new BufferGeometry();
          linkGeo.setAttribute('position', new Float32BufferAttribute(linkPos, 3));
          linkGeo.setAttribute('color', new Float32BufferAttribute(linkCol, 3));
          const linkMat = new LineBasicMaterial({
            vertexColors: true,
            transparent: true,
            opacity: 0.92,
            depthTest: true,
          });
          const linkLines = new LineSegments(linkGeo, linkMat);
          linkLines.renderOrder = 5;
          featureLinksGroup.add(linkLines);
        }

        // Subtle constraint pull-in vectors showing raw -> constrained displacement
        if (pullPos.length > 0) {
          const pullGeo = new BufferGeometry();
          pullGeo.setAttribute('position', new Float32BufferAttribute(pullPos, 3));
          pullGeo.setAttribute('color', new Float32BufferAttribute(pullCol, 3));
          const pullMat = new LineBasicMaterial({
            vertexColors: true,
            transparent: true,
            opacity: 0.34,
            depthTest: true,
          });
          const pullLines = new LineSegments(pullGeo, pullMat);
          pullLines.renderOrder = 4;
          featureLinksGroup.add(pullLines);
        }

        // Instanced observation endpoint nodes along consecutive-frame chains
        const maxNodes = Math.min(links.length * 2, 480);
        if (maxNodes > 0) {
          const nodeMesh = new InstancedMesh(endpointNodeGeo, endpointNodeMat, maxNodes);
          let nodeIdx = 0;
          const stride = Math.max(1, Math.ceil((links.length * 2) / maxNodes));
          for (let i = 0; i < links.length && nodeIdx + 1 < maxNodes; i += stride) {
            const lk = links[i];
            if (!lk.inlier) {
              _anchorColor.setRGB(0.95, 0.25, 0.37);
            } else if (lk.gapMeters < 0.09) {
              _anchorColor.setRGB(0.20, 0.92, 0.65);
            } else {
              _anchorColor.setRGB(0.96, 0.64, 0.10);
            }

            _scaleVec.set(1, 1, 1);
            _vecA.set(lk.from[0], lk.from[1], lk.from[2]);
            _mat4.compose(_vecA, _quatIdentity, _scaleVec);
            nodeMesh.setMatrixAt(nodeIdx, _mat4);
            nodeMesh.setColorAt(nodeIdx, _anchorColor);
            nodeIdx++;

            _vecA.set(lk.to[0], lk.to[1], lk.to[2]);
            _mat4.compose(_vecA, _quatIdentity, _scaleVec);
            nodeMesh.setMatrixAt(nodeIdx, _mat4);
            nodeMesh.setColorAt(nodeIdx, _anchorColor);
            nodeIdx++;
          }
          nodeMesh.count = nodeIdx;
          nodeMesh.instanceMatrix.needsUpdate = true;
          if (nodeMesh.instanceColor) nodeMesh.instanceColor.needsUpdate = true;
          nodeMesh.frustumCulled = false;
          nodeMesh.renderOrder = 5;
          featureLinksGroup.add(nodeMesh);
        }

        // Instanced 3D global anchor diamonds at each multi-frame locked feature
        const anchors = res.featureAnchors;
        if (anchors.length > 0) {
          const anchorMesh = new InstancedMesh(anchorOctaGeo, anchorOctaMat, anchors.length);
          for (let i = 0; i < anchors.length; i++) {
            const anc = anchors[i];
            const s = 0.85 + Math.min(0.65, anc.observationCount * 0.04);
            _vecA.set(anc.position[0], anc.position[1], anc.position[2]);
            _scaleVec.set(s, s * 1.25, s);
            _mat4.compose(_vecA, _quatIdentity, _scaleVec);
            anchorMesh.setMatrixAt(i, _mat4);

            if (anc.inlierRatio < 0.5) {
              _anchorColor.setRGB(0.95, 0.25, 0.37);
            } else if (anc.meanGapMeters < 0.08) {
              _anchorColor.setRGB(0.16, 0.90, 0.62);
            } else {
              _anchorColor.setRGB(0.96, 0.62, 0.08);
            }
            anchorMesh.setColorAt(i, _anchorColor);
          }
          anchorMesh.instanceMatrix.needsUpdate = true;
          if (anchorMesh.instanceColor) anchorMesh.instanceColor.needsUpdate = true;
          anchorMesh.frustumCulled = false;
          anchorMesh.renderOrder = 6;
          featureLinksGroup.add(anchorMesh);
        }
      }

      // 5. Rebuild the active synthetic frame's heuristic heatmap overlay.
      disposeConfidenceOverlayGroup();
      confidenceOverlayGroup.visible = lyr.showDepthConfidence;

      if (lyr.showDepthConfidence && res.poses.length > 0 && confData.width > 0 && confData.height > 0) {
        const activePose =
          res.poses.find((p) => p.frameIndex === activeIdx) ??
          res.poses[res.poses.length - 1];

        if (activePose) {
          const tex = new DataTexture(
            confData.rgba,
            confData.width,
            confData.height,
            RGBAFormat,
          );
          tex.flipY = true;
          tex.colorSpace = SRGBColorSpace;
          tex.magFilter = LinearFilter;
          tex.minFilter = LinearFilter;
          tex.needsUpdate = true;

          const planeDist = 1.15;
          const aspect = confData.width / Math.max(1, confData.height);
          const focalNorm = Math.max(40, activePose.focalPx) / Math.max(1, confData.width);
          const planeW = Math.max(0.85, Math.min(2.1, planeDist / focalNorm));
          const planeH = planeW / aspect;

          const rig = new Group();
          rig.position.set(...activePose.position);
          _vecA.set(...activePose.lookAt);
          rig.lookAt(_vecA);

          const planeGeo = new PlaneGeometry(planeW, planeH);
          const planeMat = new MeshBasicMaterial({
            map: tex,
            transparent: true,
            opacity: 0.78,
            side: DoubleSide,
            depthWrite: false,
          });
          const heatmapPlane = new Mesh(planeGeo, planeMat);
          heatmapPlane.position.z = planeDist;
          heatmapPlane.rotation.y = Math.PI;
          heatmapPlane.renderOrder = 3;
          rig.add(heatmapPlane);

          // Crisp border frame + 4 projection rays anchoring the heatmap to the active camera
          const hw = planeW * 0.5;
          const hh = planeH * 0.5;
          const framePts = new Float32Array([
            -hw, -hh, planeDist,  hw, -hh, planeDist,
             hw, -hh, planeDist,  hw,  hh, planeDist,
             hw,  hh, planeDist, -hw,  hh, planeDist,
            -hw,  hh, planeDist, -hw, -hh, planeDist,
            0, 0, 0, -hw, -hh, planeDist,
            0, 0, 0,  hw, -hh, planeDist,
            0, 0, 0,  hw,  hh, planeDist,
            0, 0, 0, -hw,  hh, planeDist,
          ]);
          const frameGeo = new BufferGeometry();
          frameGeo.setAttribute('position', new BufferAttribute(framePts, 3));
          const frameMat = new LineBasicMaterial({
            color: 0x22d3ee,
            transparent: true,
            opacity: 0.75,
          });
          const frameLines = new LineSegments(frameGeo, frameMat);
          frameLines.renderOrder = 4;
          rig.add(frameLines);

          confidenceOverlayGroup.add(rig);
        }
      }

      requestRender();
    }

    syncSceneRef.current = syncSceneData;

    applyCameraPresetRef.current = (preset: CameraPreset) => {
      const { result: res, activeFrameIndex: idx } = latestPropsRef.current;
      if (preset === 'reset' || preset === 'free') {
        _targetPos.set(5.4, 3.8, 5.6);
        _targetLook.set(0, 0.9, 0);
      } else if (preset === 'top') {
        _targetPos.set(0.01, 8.8, 0.01);
        _targetLook.set(0, 0.4, 0);
      } else if (preset === 'frame') {
        const pose = res.poses[idx] ?? res.poses[0];
        if (pose) {
          _vecB.set(pose.lookAt[0] - pose.position[0], pose.lookAt[1] - pose.position[1], pose.lookAt[2] - pose.position[2]).normalize();
          _targetPos.set(
            pose.position[0] - _vecB.x * 0.4,
            pose.position[1] - _vecB.y * 0.4 + 0.25,
            pose.position[2] - _vecB.z * 0.4,
          );
          _targetLook.set(...pose.lookAt);
        }
      }
      cameraTweenActive = true;
      requestRender();
    };

    const onControlStart = () => {
      cameraTweenActive = false;
      setActivePreset('free');
    };
    const onControlChange = () => {
      requestRender();
    };
    const onControlEnd = () => {
      syncSceneData();
    };
    controls.addEventListener('start', onControlStart);
    controls.addEventListener('change', onControlChange);
    controls.addEventListener('end', onControlEnd);

    const resizeObserver = new ResizeObserver((entries) => {
      if (disposed || contextUnavailable || renderFailed) return;
      for (const entry of entries) {
        const w = Math.max(1, Math.round(entry.contentRect.width));
        const h = Math.max(1, Math.round(entry.contentRect.height));
        // setSize rewrites canvas.width/height and clears its pixels, even for an unchanged size.
        if (w === renderedWidth && h === renderedHeight) continue;
        renderedWidth = w;
        renderedHeight = h;
        publishRenderState('pending');
        camera.aspect = w / h;
        camera.updateProjectionMatrix();
        renderer.setSize(w, h, false);
        requestRender();
      }
    });
    resizeObserver.observe(container);

    syncSceneData();

    return () => {
      disposed = true;
      syncSceneRef.current = () => {};
      applyCameraPresetRef.current = () => {};
      cancelAnimationFrame(frameId);
      window.clearTimeout(timerId);
      if (pendingSync && !contextUnavailable) {
        (renderer.getContext() as WebGL2RenderingContext).deleteSync(pendingSync);
      }
      pendingSync = null;
      resizeObserver.disconnect();
      controls.removeEventListener('start', onControlStart);
      controls.removeEventListener('change', onControlChange);
      controls.removeEventListener('end', onControlEnd);
      controls.dispose();
      renderer.domElement.removeEventListener('webglcontextlost', onContextLost);
      renderer.domElement.removeEventListener('webglcontextrestored', onContextRestored);

      disposeConfidenceOverlayGroup();
      // Include unused shared assets and the currently attached dynamic lines/instances.
      // Sets avoid disposing shared camera/anchor assets once per mesh instance.
      const geometries = new Set<BufferGeometry>([
        baseQuad, instancedGeo, anchorOctaGeo, endpointNodeGeo,
        bodyGeo, lensGeo, frustumPyramidGeo, frustumEdgesGeo,
      ]);
      const materials = new Set<Material>([
        splatMaterial, anchorOctaMat, endpointNodeMat,
        inactiveCamMat, activeCamMat, lensMat,
        activeLineMat, inactiveLineMat, trajLineMat, rawTrajLineMat,
      ]);
      scene.traverse((object) => {
        if (object instanceof Mesh || object instanceof Line) {
          geometries.add(object.geometry);
          const items = Array.isArray(object.material) ? object.material : [object.material];
          for (const material of items) materials.add(material);
        }
        if (object instanceof InstancedMesh) object.dispose();
      });
      for (const geometry of geometries) geometry.dispose();
      for (const material of materials) material.dispose();
      environmentTarget?.dispose();
      scene.environment = null;
      scene.clear();

      renderer.dispose();
      renderer.forceContextLoss();
      if (renderer.domElement.parentNode === container) {
        container.removeChild(renderer.domElement);
      }
    };
  }, [retryKey]);

  useLayoutEffect(() => {
    syncSceneRef.current();
  }, [result, rawPoses, activeFrameIndex, colorMode, layers, depthConfidence, depthTemporalWeight]);

  useEffect(() => {
    if (!layers.showDepthConfidence) return;
    const canvas = hudCanvasRef.current;
    if (!canvas || depthConfidence.width <= 0 || depthConfidence.height <= 0) return;
    canvas.width = depthConfidence.width;
    canvas.height = depthConfidence.height;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    const imgData = ctx.createImageData(depthConfidence.width, depthConfidence.height);
    for (let i = 0; i < depthConfidence.width * depthConfidence.height; i++) {
      const o = i * 4;
      imgData.data[o] = depthConfidence.rgba[o];
      imgData.data[o + 1] = depthConfidence.rgba[o + 1];
      imgData.data[o + 2] = depthConfidence.rgba[o + 2];
      imgData.data[o + 3] = 255;
    }
    ctx.putImageData(imgData, 0, 0);
  }, [layers.showDepthConfidence, depthConfidence]);

  const handlePreset = (preset: CameraPreset) => {
    setActivePreset(preset);
    applyCameraPresetRef.current(preset);
  };

  return (
    <div className="relative w-full h-full bg-[#07090E] overflow-hidden">
      {/* Dedicated isolated WebGL mount node */}
      <div
        ref={mountRef}
        data-testid="gaussian-webgl-viewport"
        data-render-state={renderState}
        aria-busy={renderState === 'pending'}
        className="w-full h-full cursor-grab active:cursor-grabbing"
      />

      {renderState === 'pending' && !hasPresentedFrame && (
        <div role="status" className="absolute inset-0 z-10 flex items-center justify-center bg-[#07090E]/90 pointer-events-none">
          <p className="text-sm text-slate-300">正在绘制合成三维场景…</p>
        </div>
      )}

      {(renderError || contextLost) && (
        <div className="absolute inset-0 flex items-center justify-center bg-[#07090E]/90 z-20">
          <div role="alert" className="max-w-sm mx-6 text-center space-y-4">
            <p className="text-sm text-amber-400 leading-relaxed">
              {renderError ?? 'WebGL 上下文已暂停。浏览器恢复后将继续显示合成场景，也可以手动重试。'}
            </p>
            <button
              type="button"
              onClick={() => setRetryKey((value) => value + 1)}
              className="rounded-md border border-cyan-500/50 px-4 py-2 text-sm text-cyan-300 hover:bg-cyan-500/10 focus-visible:outline-2 focus-visible:outline-cyan-400"
            >
              重试三维视图
            </button>
          </div>
        </div>
      )}

      <div className="absolute bottom-3 left-3 max-w-[calc(100%-1.5rem)] rounded-md border border-[#1E293B] bg-[#0F141D] px-2.5 py-1.5 text-[10px] text-slate-400 pointer-events-none">
        合成场景 · 高斯贴片示意 · 最多显示 16,000 个
      </div>

      {/* Minimal Top-Left Camera Preset Toolbar (no backdrop-blur over WebGL) */}
      <div className="absolute top-3 left-3 z-10 flex items-center gap-1 p-1 rounded-md bg-[#0F141D] border border-[#1E293B]">
        {(
          [
            { id: 'reset', label: '重置视角' },
            { id: 'top', label: '轨迹俯视' },
            { id: 'frame', label: '跟随当前帧' },
          ] as const
        ).map((cam) => (
          <button
            key={cam.id}
            type="button"
            onClick={() => handlePreset(cam.id)}
            className={`px-2.5 py-1 rounded text-[11px] font-medium transition-colors cursor-pointer whitespace-nowrap ${
              activePreset === cam.id
                ? 'bg-cyan-500 text-cyan-950 font-semibold'
                : 'text-slate-300 hover:text-white hover:bg-[#1E293B]'
            }`}
          >
            {cam.label}
          </button>
        ))}
      </div>

      {/* Minimal Top-Right Color Scale & Cross-Frame Anchor Legend */}
      <div className="hidden sm:flex flex-col items-end gap-1.5 absolute top-3 right-3 z-10">
        <div className="flex items-center gap-3 px-3 py-1.5 rounded-md bg-[#0F141D] border border-[#1E293B] text-[11px] font-mono text-slate-300">
          {layers.showDepthConfidence ? (
            <span className="text-cyan-300">叠加: 合成深度启发式热图</span>
          ) : (
            <>
              {colorMode === 'rgb' && (
                <span>着色: 合成 RGB 外观</span>
              )}
              {colorMode === 'depth' && (
                <div className="flex items-center gap-2">
                  <span>合成深度:</span>
                  <span className="text-amber-300">近 1.2m</span>
                  <span
                    aria-hidden="true"
                    className="w-16 h-2 rounded-xs bg-gradient-to-r from-amber-400 via-cyan-400 to-blue-900"
                  />
                  <span className="text-blue-300">远 7.7m</span>
                </div>
              )}
              {colorMode === 'error' && (
                <div className="flex items-center gap-2">
                  <span>演示残差:</span>
                  <span className="text-cyan-400">0.00m</span>
                  <span
                    aria-hidden="true"
                    className="w-16 h-2 rounded-xs bg-gradient-to-r from-cyan-400 via-slate-400 to-amber-500"
                  />
                  <span className="text-amber-400">&ge;0.22m</span>
                </div>
              )}
            </>
          )}
        </div>

        {layers.showFeatureLinks && (
          <div
            data-testid="feature-links-hud"
            className="flex items-center gap-2.5 px-3 py-1.5 rounded-md bg-[#0F141D] border border-[#1E293B] text-[11px] font-mono tabular-nums text-slate-300"
          >
            <span className="text-emerald-400">合成锚点连线</span>
            <span>{result.metrics.inlierLinkCount} 对</span>
            <span aria-hidden="true" className="text-slate-600">·</span>
            <span>
              演示间距{' '}
              <strong
                data-testid="feature-gap-mm"
                className={
                  result.metrics.meanFeatureGapMm < 80
                    ? 'text-emerald-400'
                    : 'text-amber-400'
                }
              >
                {result.metrics.meanFeatureGapMm.toFixed(1)} mm
              </strong>
            </span>
          </div>
        )}

        {layers.showDepthConfidence && (
          <div
            data-testid="depth-confidence-hud"
            className="w-52 p-2.5 rounded-md bg-[#0F141D] border border-cyan-500/40 text-[11px] font-mono tabular-nums text-slate-300 flex flex-col gap-2"
          >
            <div className="flex items-center justify-between gap-2">
              <span className="text-cyan-300 font-semibold">
                启发式热图 · F{String(activeFrameIndex + 1).padStart(2, '0')}
              </span>
              <span
                data-testid="depth-confidence-mean"
                className={
                  depthConfidence.meanConfidence >= 0.72
                    ? 'text-emerald-400 font-semibold'
                    : 'text-amber-400 font-semibold'
                }
              >
                {depthConfidence.meanConfidence.toFixed(2)}
              </span>
            </div>

            <canvas
              ref={hudCanvasRef}
              aria-label="当前合成帧启发式分数热力图"
              className="w-full aspect-[160/96] rounded border border-[#1E293B] bg-[#07090E] block"
            />

            <div className="flex items-center gap-1.5 text-[10px]">
              <span className="text-rose-400">风险</span>
              <span
                aria-hidden="true"
                className="flex-1 h-1.5 rounded-xs bg-gradient-to-r from-rose-500 via-amber-400 via-cyan-400 to-emerald-400"
              />
              <span className="text-emerald-400">稳定</span>
            </div>

            <div className="flex items-center justify-between text-[10px] text-slate-400">
              <span>高分区域 {Math.round(depthConfidence.highConfidenceRatio * 100)}%</span>
              <span>低分区域 {Math.round(depthConfidence.lowConfidenceRatio * 100)}%</span>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
