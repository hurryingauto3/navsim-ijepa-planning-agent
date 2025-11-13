"use client";

export default function AboutPage() {
  return (
    <div className="min-h-screen">
      <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        <div className="space-y-8">
          {/* Header */}
          <div className="text-center space-y-4">
            <h1 className="text-4xl font-bold text-white">SSL Autonomous Vehicle Planning with I-JEPA</h1>
            <p className="text-xl text-neutral-400">
              Autonomous Vehicle Planning with Self-Supervised Learning
            </p>
            <p className="text-sm text-neutral-500">
              Master&apos;s Thesis | NYU Tandon School of Engineering | 2025-2026
            </p>
          </div>

          {/* Overview */}
          <section className="space-y-4">
            <h2 className="text-2xl font-semibold text-white border-b border-neutral-800 pb-2">
              Project Overview
            </h2>
            <div className="prose prose-invert max-w-none space-y-4 text-neutral-300">
              <p>
                This project demonstrates a novel approach to autonomous vehicle trajectory planning
                that achieves state-of-the-art performance using <strong>90% less labeled training data</strong>.
                By leveraging self-supervised learning with pre-trained vision models, we show that
                effective planning can be learned with minimal supervision.
              </p>
              <p>
                The system processes camera inputs to predict safe, efficient trajectories for autonomous
                vehicles navigating complex urban environments. The key innovation is the use of
                pre-trained visual representations that already understand driving scenes, requiring
                only a lightweight planning head to be trained on a small subset of labeled data.
              </p>
            </div>
          </section>

          {/* Key Results */}
          <section className="space-y-4">
            <h2 className="text-2xl font-semibold text-white border-b border-neutral-800 pb-2">
              Key Results
            </h2>
            <div className="grid md:grid-cols-3 gap-6">
              <div className="bg-neutral-900 rounded-lg p-6 border border-neutral-800">
                <div className="text-3xl font-bold text-green-400 mb-2">90%</div>
                <div className="text-sm text-neutral-400">
                  Reduction in labeled data requirements compared to baseline methods
                </div>
              </div>
              <div className="bg-neutral-900 rounded-lg p-6 border border-neutral-800">
                <div className="text-3xl font-bold text-blue-400 mb-2">0.8466</div>
                <div className="text-sm text-neutral-400">
                  PDMS score achieved using only 10% of training data
                </div>
              </div>
              <div className="bg-neutral-900 rounded-lg p-6 border border-neutral-800">
                <div className="text-3xl font-bold text-purple-400 mb-2">5%</div>
                <div className="text-sm text-neutral-400">
                  Performance gap from state-of-the-art using full dataset
                </div>
              </div>
            </div>
          </section>

          {/* Technical Approach */}
          <section className="space-y-4">
            <h2 className="text-2xl font-semibold text-white border-b border-neutral-800 pb-2">
              Technical Approach
            </h2>
            <div className="space-y-4 text-neutral-300">
              <div className="bg-neutral-900 rounded-lg p-6 border border-neutral-800">
                <h3 className="text-lg font-semibold text-white mb-3">Architecture</h3>
                <ul className="space-y-2 list-disc list-inside text-sm">
                  <li>
                    <strong>Pre-trained Vision Encoder:</strong> Large-scale self-supervised model
                    (630M parameters) that understands driving scenes without labeled data
                  </li>
                  <li>
                    <strong>Lightweight Planning Head:</strong> Small MLP (500K parameters) that
                    maps visual features to trajectory waypoints
                  </li>
                  <li>
                    <strong>Training Strategy:</strong> Only the planning head is trained, keeping
                    the encoder frozen to preserve learned representations
                  </li>
                </ul>
              </div>
              <div className="bg-neutral-900 rounded-lg p-6 border border-neutral-800">
                <h3 className="text-lg font-semibold text-white mb-3">Why This Works</h3>
                <p className="text-sm">
                  Pre-trained vision models have already learned to recognize vehicles, pedestrians,
                  lanes, and traffic patterns from large-scale unlabeled image datasets. By reusing
                  these representations, we can focus training on the much simpler task of mapping
                  scene understanding to driving actions, dramatically reducing data requirements.
                </p>
              </div>
            </div>
          </section>

          {/* Engineering Highlights */}
          <section className="space-y-4">
            <h2 className="text-2xl font-semibold text-white border-b border-neutral-800 pb-2">
              Engineering Highlights
            </h2>
            <div className="grid md:grid-cols-2 gap-4">
              <div className="bg-neutral-900 rounded-lg p-4 border border-neutral-800">
                <h3 className="font-semibold text-white mb-2">Scalable Training</h3>
                <p className="text-sm text-neutral-400">
                  Multi-node distributed training (DDP) across 4 nodes × 4 GPUs, with automatic
                  checkpointing and recovery for production-grade reliability.
                </p>
              </div>
              <div className="bg-neutral-900 rounded-lg p-4 border border-neutral-800">
                <h3 className="font-semibold text-white mb-2">Performance Optimization</h3>
                <p className="text-sm text-neutral-400">
                  Mixed-precision training (FP16) for 2× speedup, optimized data pipelines, and
                  efficient memory management for large-scale experiments.
                </p>
              </div>
              <div className="bg-neutral-900 rounded-lg p-4 border border-neutral-800">
                <h3 className="font-semibold text-white mb-2">Web-Based Demo</h3>
                <p className="text-sm text-neutral-400">
                  Interactive browser showcase with real-time visualization, supporting both
                  cached replays and live inference for easy demonstration and evaluation.
                </p>
              </div>
              <div className="bg-neutral-900 rounded-lg p-4 border border-neutral-800">
                <h3 className="font-semibold text-white mb-2">MLOps Integration</h3>
                <p className="text-sm text-neutral-400">
                  Experiment tracking, model versioning, and deployment pipelines for reproducible
                  research and production readiness.
                </p>
              </div>
            </div>
          </section>

          {/* Demo Instructions */}
          <section className="space-y-4">
            <h2 className="text-2xl font-semibold text-white border-b border-neutral-800 pb-2">
              Try the Demo
            </h2>
            <div className="bg-neutral-900 rounded-lg p-6 border border-neutral-800">
              <p className="text-neutral-300 mb-4">
                The interactive demo on the home page allows you to:
              </p>
              <ul className="space-y-2 list-disc list-inside text-sm text-neutral-400">
                <li>Select different planning models and compare their performance</li>
                <li>Visualize trajectories in real-time using WebGL rendering</li>
                <li>View metrics such as EPDMS (End-to-End Planning Driving Metric Score)</li>
                <li>Replay cached simulations or run live inference</li>
              </ul>
              <div className="mt-4">
                <a
                  href="/"
                  className="inline-block px-4 py-2 bg-white text-black rounded-md font-medium hover:bg-neutral-200 transition"
                >
                  Go to Demo →
                </a>
              </div>
            </div>
          </section>

          {/* Footer */}
          <div className="text-center text-sm text-neutral-500 pt-8 border-t border-neutral-800">
            <p>
              This project demonstrates research capabilities in computer vision, deep learning,
              and autonomous systems. For questions or collaboration opportunities, please reach out.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}

