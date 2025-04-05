import React from "react";
import "@fontsource/audiowide";
import "@fontsource/source-code-pro";
// eslint-disable-next-line no-unused-vars
import { motion } from "framer-motion"; 
import './styles/globals.css';

export default function HomePage() {
  return (
      <main className="min-h-screen bg-black text-white font-Audiowide tracking-wider">
        <section
          className="relative h-[500px] bg-center bg-cover bg-no-repeat flex items-center justify-center"

          style={{ 
            backgroundImage: "url('/TKYObg.png')",
            backgroundSize: "cover",
            backgroundPosition: "center",
          }}
        >
          <div className="absolute inset-0"></div>
          <div className="relative flex flex-col items-center text-center mx-auto">

            <motion.h2
              className="bg-black border-indigo-800 font-Audiowide rounded justify-items-center p-2 tracking-tight my-1 mx-auto max-w-xl text-center hover:bg-techviolet hover:text-gray-900 transition shadow-xl"
              initial={{ opacity: 0, y: -20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 1.9 }}
              style={{
                textShadow: "0 0 10px #B94EFF",
                boxShadow: "0 0 10px #B94EFF",
              }}
            >
              AI <span className="font-bold text-xl text-gray-100" style={{ textShadow: "0 0 9px #B94EFF" }}>T</span>emporal{" "}
              <span className="font-bold text-xl text-gray-100" style={{ textShadow: "0 0 9px #B94EFF" }}>K</span>nowledge{" "}
              <span className="font-bold text-xl text-gray-100" style={{ textShadow: "0 0 9px #B94EFF" }}>Y</span>ield{" "}
              <span className="font-bold text-xl text-gray-100" style={{ textShadow: "0 0 9px #B94EFF" }}>O</span>utput Drift Analyzer
            </motion.h2>

            <motion.p
              className="text-lg justify-items-center"
              style={{
                textShadow: "0 0 12px rgba(255, 255, 255, 0.2)",
                boxShadow: "0 0 10px neonpurple",
              }}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 1.2, duration: 1.32 }}
            >
            </motion.p>
          </div> 
          
          {/* <div>
            <motion.h1
              className="text-fuchsia-500 text-9xl md:text-9xl font-extralight mb-6 tracking-widest font-['Audiowide'] uppercase relative z-10"
              style={{
                textShadow: "0 0 10px #ff4de8, 0 0 10px #ff4de8, 0 0 10px #ff4de8",
              }}
              initial={{ opacity: 0, y: 40 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.8, duration: 1.9 }}
            >
              TKYO DRIFT
              <span className="inline-block w-2 h-9 bg-[#833ff8] ml-8 animate-pulse"></span>
            </motion.h1>
          </div> */}

          <div className="mx-auto flex flex-col items-center gap-8 tracking-wide">
            {/* <button
              className="bg-black border border-purple-700 w-80 mx-auto font-mono text-xl tracking-tight text-white px-2 py-2 rounded hover:bg-techViolet hover:text-gray-700 transition duration-300 shadow-xl"
              style={{
                boxShadow: "0 0 10px #7C3AED",
                textShadow: "0 10 12px neonpurple",
              }}
            >
              ./get-started with npm
            </button> */}
       
            {/* <button
              className="bg-black border border-purple-700 w-75 mx-auto font-mono text-md tracking-tight text-white px-2 py-2 rounded hover:bg-techviolet hover:text-gray-700 transition duration-300   shadow-lg"
              style={{
                boxShadow: "0 0 10px #7C3AED",
                textShadow: "0 0 12px neonpurple",
              }}
            >
              ./contribute on github
            </button>           */}
          </div>
        </section>

        <section 
          className="relative py-24 px-6 bg-black overflow-hidden text-center"
          style={{ backgroundImage: "url('/cyberbg.png')" }}
        >
        <div className="absolute inset-0 bg-black black opacity-90 z-0" />

          <motion.div
            className="relative z-15 max-w-4xl mx-auto"
            initial={{ opacity: 0, y: 40 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 1.2 }}
            viewport={{ once: true }}
          >
            <h2
              className="text-5xl font-Audiowide text-neonPurple mb-3 uppercase tracking-widest font-semibold ≈"
              style={{ textShadow: "0 0 4px #B94EFF, 0 0 10px #B94EFF" }}
            >
              The Problem
            </h2>

            <p className="text-md leading-tighter text-white px-4 font-Audiowide" style={{ textShadow: "0 0 8px neonpurple" }}>
              Drift occurs when a user’s input or a language model’s output differs from what we would expect, given the data used to train that model.
              <br />
              <br />
              When your AI model receives an update and starts responding differently than you trained it to, or when users begin to ask questions that don’t quite line up with the ones you initially fed your model,
              it can be indicative of a meaningful change that requires your model to be retrained, or it can be a one-off instance that does constitute drift, but does not represent a change that requires your action at that time.
              <br />
              <br />
              <span className="text techViolet" >TKYO Drift</span> logs each pair of inputs and outputs and gives a score relative to both the training data you input and a rolling set of data that we start tracking with your first query.
              <br />
              <br />
              By comparing new inputs and outputs against these, we will be able to detect drift instantially, as well as cumulatively.
              Instances of drift can be helpful on their own, but as patterns emerge, you will get a clearer answer to the question “is my model drifting
            </p>
          </motion.div>
        </section>
    
        <section className="bg-black text-white py-8 px-4">
          <div className="max-w-7xl mx-auto grid md:grid-cols-2 gap-10 items-center">
    
    {/* Left side: text content */}
          <div className="text-center space-x-0">
            <h2
              className="text-4xl font-Audiowide text-neonPurple uppercase tracking-widest px-2 py-2"
              style={{ textShadow: "0 0 7px #B94EFF" }}
            >
              See Your Drift
            </h2>
            <p className="text-md text-gray-300 font-exralight tracking-tightest font-code py-2 display-flex-center">
              Automatically siphon I/O from any text-based AI workflow and detect drift.
              <br /><br />
              AI Model Agnostic
              <br /><br />
              Integrated embedding model within Repo
              <br /><br />
              CLI tool & Integrated logging
            </p>
          </div>

         
    
          <div className="text-center space-x-0">
            <h2
              className="text-4xl font-Audiowide text-neonPurple px-2 py-2 uppercase tracking-widest"
              style={{ textShadow: "0 0 12px #463aed" }}
            >
              Without Overhead
            </h2>
              <p className="text-md text-gray-300 font-exralight tracking-tightest font-code py-2 display-flex-center">
                HNSW of K nearest neighbors
                <br /><br />
                Ensure subclustering accounted for 
                <br /><br />
                Minimal workflow speed incursion
                <br /><br /> 
                Binary file storage for minimal space
              </p>
            </div>
          </div>
        </section>


       
{/* 
        <div className="bg-gradient-to-b from-gray-900 to-black p-4 py-10 rounded-lg shadow-inner border border-gray-800 font-mono text-left text-gray-200">
          <p className="text-techViolet mb-2 text-sm">
            analyzeInputDrift(userQuery)
          </p>
          <p>
            Input Cosine Similarity: <span className="text-green-400">.83 ✅</span>
          </p>
          <p>
            Output Cosine Similarity: <span className="text-red-500">.64 ❌</span>
          </p>
        </div> */}

       <section>       
        <div
          className="py-10 px-10 bg-gradient-to-b from-gray-800 to-black font-code text-indigo-600 font-md max-w-7xl mx-auto grid md:grid-cols-3 gap-12">
          {[
            {
              title: "Automated AI Intercept",
              content: "Model agnostic intercept of inputs & outputs using function hooks.",
            },
            {
              title: "Baseline Comparison",
              content: "Compare vector baselines to inputs & outputs to detect drift.",
            },
            {
              title: "CLI Tools & Exportable Logs",
              content:
                "Exportable logs to integrate with external visualization tools. Integrated CLI tool for spot checking drift on demand.",
            },
          ].map((feature, idx) => (
            <div
              key={idx}
              className="bg-black border border-gray-600 p-2 py-6 rounded-xl shadow-lg hover:shadow-lg hover:shadow-[#463aed] transition"
            >
              <h3
                className="text-xl text-center font-Audiowide text-indigo-500 mb-6 tracking-wider font-bold uppercase"
              >
                {feature.title}
              </h3>
                <p
                  className="text-center font-Audiowide text-gray-400 tracking-wide justify-items-center"
                >
                  {feature.content}
                </p>
            </div>
          ))}
        </div>
      </section>
  
      <section className="bg-black text-white py-4 px-3 border border-gray-800">
  <div className="max-w-5xl mx-auto text-center">
    <h2
      className="text-4xl font-bold font-Audiowide text-[rgb(185,78,255)] tracking-widest uppercase mb-4 my-1"
      style={{ textShadow: "0 0 18px #B94EFF" }}
    >
      Seamless Interface
    </h2>
    <p
      className="text-md text-gray-300 font-mono tracking-tighter max-4xl"
      style={{ textShadow: "0 0 4px rgba(255,255,255,0.2)" }}
    >
      Designed with intuitive workflows in mind, TKYO Drift blends powerful insights with a sleek, developer-first interface. 
      Focus on what matters while we handle the detection.
    </p>

    <div className="mt-4 mb-2 grid md:grid-cols-4 gap-10 tracking-wide font-bold">
      {[
        {
          title: "Informed Architecture",
          description: "HNSWlib Architecture and Binary File Storage with custom bin file reader.",
        },
        {
          title: "Comprehensive Analysis",
          description: "Support I/O conceptual/semantic drift across 2 baselines for 8 drift.",
        },
        {
          title: "Minimal Setup & Simple Integration",
          description: "Single function call generates drift scores with no config nightmares.",
        },
        {
          title: "No Databases Required",
          description: "Optimized for speed & usable in production.  Yes. Production."
        }
      ].map((item, idx) => (
        <div
          key={idx}
          className="bg-gradient-to-b from-gray-900 to-black border border-gray-700 p-6 rounded-xl shadow-md hover:shadow-[#B94EFF]/50 transition"
        >
          <h3 className="text-lg font-Audiowide text-indigo-600 mb-2 tracking-wider uppercase">{item.title}</h3>
          <p className="text-sm text-gray-400 font-code">{item.description}</p>
        </div>
      ))}
    </div>
  </div>
</section>

  
      {/* <section className="py-20 px-6 text-center bg-black border border-gray-800 font-source-code-pro">
        <h2 className="text-2xl md:text-2xl font-bold text-[rgb(109,24,255)] mb-2 tracking-widest">
          Real Time Monitoring
        </h2>
        <p className="mb-4 text-white">
          No setup hassle. Single function call generates drift scores.
        </p>
        <button
          className="bg-[#7C3AED] text-white px-6 py-2 text-lg font-bold rounded-full shadow-md hover:bg-violet-500 transition"
          style={{ boxShadow: "0 0 10px #7C3AED" }}
        >
          Monitor your drift now!
        </button>
      </section> */}
  
      <section>
        <div className="mx-auto flex flex-col items-center gap-8 tracking-wide">
          <button
            className="bg-black border border-purple-700 w-80 mx-auto font-mono text-xl tracking-tight text-white px-2 py-2 rounded hover:bg-techViolet hover:text-gray-700 transition duration-300 shadow-xl"
            style={{
              boxShadow: "0 0 10px #7C3AED",
              textShadow: "0 10 12px neonpurple",
            }}
          >
            ./get-started with npm
          </button>

          <button
              className="bg-black border border-purple-700 w-75 mx-auto font-mono text-md tracking-tight text-white px-2 py-2 rounded hover:bg-techviolet hover:text-gray-700 transition duration-300   shadow-lg"
              style={{
                boxShadow: "0 0 10px #7C3AED",
                textShadow: "0 0 12px neonpurple",
              }}
            >
              ./contribute on github
            </button> 
        </div>
      </section>
      
  
    </main>
  )
}